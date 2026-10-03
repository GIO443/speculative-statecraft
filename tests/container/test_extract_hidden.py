"""Runs in the trainer container (needs torch + transformers); skipped natively."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from spec.extract_hidden import (  # noqa: E402
    common_prefix_len,
    encode,
    extract_game,
    group_by_turn,
    turn_hidden,
    usable,
)

VOCAB = 64
STOP = 63


def tiny_qwen2() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return transformers.Qwen2Model(cfg).eval()


class CharTokenizer:
    """Maps characters to ids; the chat template is a plain join, enough to exercise encode()."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<a>"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) % (VOCAB - 1) for c in text]}


def test_common_prefix_len() -> None:
    assert common_prefix_len([[1, 2, 3, 4], [1, 2, 5]]) == 2
    # Identical sequences keep one token each.
    assert common_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2
    assert common_prefix_len([[7], [7, 8]]) == 0


def test_prefix_reuse_matches_full_forward() -> None:
    model = tiny_qwen2()
    seqs = [
        [1, 2, 3, 4, 5, 6, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 20, 21],
        [1, 2, 3, 4, 5, 6, 30, 31, 32, 33, 34],
    ]
    p = common_prefix_len(seqs)
    assert p == 6
    prefix_h, suffix_h = turn_hidden(model, seqs, p, torch.device("cpu"))
    for seq, h in zip(seqs, suffix_h, strict=True):
        with torch.inference_mode():
            full = model(input_ids=torch.tensor([seq])).last_hidden_state[0]
        torch.testing.assert_close(prefix_h, full[:p], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(h, full[p:], rtol=1e-4, atol=1e-5)


def _row(turn: int, actor: str, fid: int | None, user: str, completion: str, **kw) -> dict:
    row = {
        "game_seed": 100,
        "n_factions": 2,
        "turn": turn,
        "actor": actor,
        "faction": fid,
        "messages": [
            {"role": "system", "content": f"shared state {turn}"},
            {"role": "user", "content": user},
        ],
        "completion": completion,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": "stop",
        "guided": actor == "faction",
        "valid_json": True if actor == "faction" else None,
        "legal": True if actor == "faction" else None,
        "error": None,
    }
    return row | kw


def test_extract_game_layout() -> None:
    rows = [
        _row(0, "faction", 0, "you are 0", '{"a":1}'),
        _row(0, "faction", 1, "you are 1", '{"b":2}'),
        _row(0, "narrator", None, "narrate", "It rained."),
        _row(0, "faction", 1, "you are 1", "", error="APIError: boom"),
        _row(1, "faction", 0, "you are 0", '{"c":3}', finish_reason="length"),
    ]
    assert [usable(r) for r in rows] == [True, True, True, False, True]
    assert list(group_by_turn(rows)) == [0, 1]

    tok = CharTokenizer()
    tensors, samples, stats = extract_game(
        tiny_qwen2(), tok, STOP, rows, torch.device("cpu"), torch.bfloat16
    )
    assert stats["samples"] == 4 and stats["skipped"] == 1
    assert [s["k"] for s in samples] == [0, 1, 2, 3]
    for s in samples:
        t = s["turn"]
        ids = torch.cat([tensors[f"t{t}.prefix_ids"], tensors[f"s{s['k']}.ids"]]).tolist()
        assert len(ids) == s["length"]
        hidden = torch.cat([tensors[f"t{t}.prefix_hidden"], tensors[f"s{s['k']}.hidden"]])
        assert hidden.shape == (s["length"], 32) and hidden.dtype == torch.bfloat16
        # Stopped completions end with the stop token; truncated ones do not.
        assert (ids[-1] == STOP) == (s["finish_reason"] == "stop")
        assert s["completion_start"] < s["length"]
    # The turn-0 prefix covers the shared system message, so it is stored once for 3 samples.
    p0 = samples[0]["prefix_len"]
    assert p0 >= len("<system>shared state 0<user>")
    assert stats["tokens_stored"] < stats["tokens_total"]


def test_encode_prefers_server_ids() -> None:
    tok = CharTokenizer()
    row = _row(0, "faction", 0, "you are 0", "ab")
    retok = encode(tok, row, STOP)
    assert retok.ids[-3:] == [ord("a") % (VOCAB - 1), ord("b") % (VOCAB - 1), STOP]

    # Server sampled "ab" as one non-canonical token (5) and its prompt as the canonical ids.
    prompt = retok.ids[: retok.completion_start]
    server = row | {"prompt_token_ids": prompt, "token_ids": [5, STOP]}
    e = encode(tok, server, STOP)
    assert e.ids == [*prompt, 5, STOP]
    assert e.completion_start == len(prompt)
    assert e.prompt_matches is True and e.completion_matches is False
