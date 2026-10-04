"""Training math and the exact chained-draft evaluation. Trainer container only."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_extract_hidden import STOP, CharTokenizer, _row, tiny_qwen2  # noqa: E402

from spec.draft_head import DraftHead, HeadConfig  # noqa: E402
from spec.extract_hidden import extract_game, write_shard  # noqa: E402
from spec.train import (  # noqa: E402
    SampleRef,
    TrainConfig,
    chain_accept,
    head_inputs,
    index_shards,
    load_sample,
    lr_at,
    sample_loss,
    split,
    token_regions,
)

H, V = 32, 64
CFG = HeadConfig(
    hidden_size=H,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    rms_norm_eps=1e-6,
    rope_theta=10_000.0,
    vocab_size=V,
    max_position_embeddings=256,
)


def _train_cfg(**kw) -> TrainConfig:
    base = dict(
        name="t", target="x", hidden_dir="h", runs_dir="r", heads_dir="d", results_dir="o",
        val_seed_mod=10, val_seed_rem=9, epochs=1, lr=1e-3, min_lr_ratio=0.1, warmup_steps=10,
        weight_decay=0.0, grad_clip=1.0, accum=1, feature_noise=0.0, w_reg=1.0, w_cls=0.1,
        eval_depth=3, eval_every_steps=10, eval_samples=4, log_every_steps=1, num_workers=0,
        seed=0,
    )  # fmt: skip
    return TrainConfig(**(base | kw))


def _setup(n: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    head = DraftHead(CFG).eval()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.2)
    emb = torch.randn(V, H)
    ids = torch.randint(0, V, (n,))
    feats = torch.randn(n, H)
    return head, emb, ids, feats


def test_forward_queries_equals_full_forward() -> None:
    head, emb, ids, feats = _setup()
    with torch.no_grad():
        x, pos = head_inputs(head, emb, ids, feats)
        full = head.forward(emb[ids[1:]][None], feats[:-1][None], pos[None])
        k, v = head.context(x, pos[None])
        q = torch.tensor([3, 7, 10])
        part = head.forward_queries(x[:, q], q[None], k, v, pos[None, :] <= q[:, None])
    torch.testing.assert_close(part, full[:, q], rtol=1e-5, atol=1e-5)


def test_chain_depth2_matches_explicit_recompute() -> None:
    """Step 2 at anchor a == full causal pass over target inputs 0..a plus the draft input
    (embed(t1), out1) at position a+1, which is what vLLM computes."""
    head, emb, ids, feats = _setup(n=10)
    c = 4
    with torch.no_grad():
        _, pos = head_inputs(head, emb, ids, feats)
        out_full = head.forward(emb[ids[1:]][None], feats[:-1][None], pos[None])[0]
        for a in range(c - 1, len(ids) - 2):
            t1 = (out_full[a] @ emb.T).argmax()
            emb_seq = torch.cat([emb[ids[1 : a + 2]], emb[t1][None]])
            feat_seq = torch.cat([feats[: a + 1], out_full[a][None]])
            ref = head.forward(emb_seq[None], feat_seq[None], torch.arange(a + 2)[None])[0, -1]
            t2_ref = (ref @ emb.T).argmax()
            # Force a "hit" at depth 1 so the chain reaches depth 2, then compare tokens.
            ids_hit = ids.clone()
            ids_hit[a + 2] = t1
            ids_hit[a + 3 if a + 3 < len(ids) else a + 2] = t2_ref if a + 3 < len(ids) else t1
            anchors, acc = chain_accept(head, emb, ids_hit, feats, c, depth=2)
            i = (anchors == a).nonzero().item()
            if a + 3 < len(ids):
                assert acc[i].item() == 2, a
            else:
                assert acc[i].item() >= 1, a


def test_chain_counts_leading_matches_only() -> None:
    head, emb, ids, feats = _setup(n=10)
    with torch.no_grad():
        anchors, acc = chain_accept(head, emb, ids, feats, 4, depth=3)
    assert anchors.tolist() == list(range(3, 8))
    assert ((acc >= 0) & (acc <= 3)).all()
    # Destroy every depth-1 target: nothing can be accepted.
    with torch.no_grad():
        out = head.forward(emb[ids[1:]][None], feats[:-1][None], torch.arange(9)[None])[0]
    first = (out @ emb.T).argmax(-1)
    bad = ids.clone()
    for a in anchors.tolist():
        bad[a + 2] = (first[a] + 1) % V
    _, acc_bad = chain_accept(head, emb, bad, feats, 4, depth=3)
    assert (acc_bad == 0).all()


def test_sample_loss_trains() -> None:
    head, emb, ids, feats = _setup(n=16)
    head.train()
    cfg = _train_cfg()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        loss, st = sample_loss(head, emb, ids, feats, 6, cfg)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert st["positions"] == 16 - 1 - (6 - 1)
    assert loss.item() < 0.5 * first


def test_token_regions() -> None:
    class Tok:
        def decode(self, ids: list[int]) -> str:
            return "".join(chr(i) for i in ids)

    text = '{"action": {"type": "pass"}, "diplomatic_message": "Hi all"}'
    ids = [ord(ch) for ch in text]
    regions = token_regions(Tok(), ids, "faction")
    msg = "".join(ch for ch, r in zip(text, regions, strict=True) if r == "message")
    assert msg == "Hi all"
    assert token_regions(Tok(), ids, "narrator") == ["narrator"] * len(ids)


def test_split_and_schedule() -> None:
    refs = [SampleRef("s", 0, 0, seed, 4, "faction", 1, 2) for seed in range(100, 120)]
    train, val = split(refs, 10, 9)
    assert sorted(r.game_seed for r in val) == [109, 119] and len(train) == 18
    cfg = _train_cfg()
    assert lr_at(0, 100, cfg) == pytest.approx(0.1)
    assert lr_at(10, 100, cfg) == pytest.approx(1.0)
    assert lr_at(100, 100, cfg) == pytest.approx(0.1)


def test_index_and_load_roundtrip(tmp_path: Path) -> None:
    rows = [
        _row(0, "faction", 0, "you are 0", '{"a":1}'),
        _row(0, "faction", 1, "you are 1", '{"b":2}'),
        _row(1, "narrator", None, "narrate", "It rained."),
    ]
    tensors, samples, _ = extract_game(
        tiny_qwen2(), CharTokenizer(), STOP, rows, torch.device("cpu"), torch.bfloat16
    )
    write_shard(tmp_path / "n2_s100.safetensors", tensors, {"samples": samples})
    refs = index_shards(tmp_path)
    assert [(r.k, r.turn, r.actor) for r in refs] == [
        (0, 0, "faction"),
        (1, 0, "faction"),
        (2, 1, "narrator"),
    ]
    for r in refs:
        ids, feats = load_sample(r)
        assert len(ids) == len(feats) == r.length and ids.dtype == torch.int64
