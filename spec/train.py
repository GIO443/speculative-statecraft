"""Phase 3 (trainer container): train the EAGLE-1 draft head on extracted hidden states.

    docker compose --profile train run --rm trainer \
        python -m spec.train configs/train/<name>.yaml [--max-steps N] [--eval-samples N]

Per sample (one game request) with token ids x_0..x_{n-1}, target features f_0..f_{n-1} and
completion start c, the head runs at positions i = 0..n-2 on (embed(x_{i+1}), f_i). Only
positions that matter at serving time carry loss: i >= c-1, where the head predicts completion
tokens (the first completion token is sampled by the target itself). Losses, as in EAGLE-1:

    reg = smooth_l1(out_i, f_{i+1})                                  feature regression
    cls = CE(softmax(f_{i+1} E^T), log_softmax(out_i E^T))           target's next-token dist.

with E the target's (tied) embedding / LM head, frozen. Inputs get uniform feature noise.

Evaluation chains `eval_depth` greedy draft steps exactly as vLLM drafts them: step 1 uses the
target feature, later steps feed the head's own output back with the drafted token, attending
to target-feature keys up to the anchor plus the earlier draft steps' keys. A drafted token
counts as accepted when it equals the logged token. The logs were sampled from the target's own
(guided, temperature) distribution, so P(match) is exactly the rejection sampler's acceptance
probability for a greedy draft, and the chained match length estimates vLLM's accepted length.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field
from safetensors import safe_open

from spec.draft_head import DraftHead, HeadConfig
from spec.export import export

REPO_ROOT = Path(__file__).resolve().parents[1]
MESSAGE_VALUE = re.compile(r'"diplomatic_message"\s*:\s*"((?:[^"\\]|\\.)*)')


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    target: str  # HF id of the target model (embedding / LM head, tokenizer, head shape)
    hidden_dir: str  # extract_hidden output
    runs_dir: str  # checkpoints (large; Docker volume)
    heads_dir: str  # exported heads, mounted into vLLM
    results_dir: str  # small logs, relative to the repo
    # Held-out games: seed % val_seed_mod == val_seed_rem (10% of every faction count).
    val_seed_mod: int = Field(ge=2)
    val_seed_rem: int = Field(ge=0)
    epochs: int = Field(ge=1)
    lr: float
    min_lr_ratio: float  # cosine floor as a fraction of lr
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    accum: int = Field(ge=1)  # samples per optimizer step
    feature_noise: float  # uniform(-a, a) added to input features
    w_reg: float
    w_cls: float
    eval_depth: int = Field(ge=1)
    eval_every_steps: int
    eval_samples: int  # val samples per periodic eval (the final eval uses all)
    log_every_steps: int
    num_workers: int
    seed: int


@dataclass(frozen=True)
class SampleRef:
    shard: str
    k: int
    turn: int
    game_seed: int
    n_factions: int
    actor: str
    completion_start: int
    length: int


def index_shards(hidden_dir: Path) -> list[SampleRef]:
    refs = []
    for path in sorted(hidden_dir.glob("*.safetensors")):
        with safe_open(str(path), "pt") as f:
            samples = json.loads(f.metadata()["samples"])
        refs.extend(
            SampleRef(
                shard=str(path),
                k=s["k"],
                turn=s["turn"],
                game_seed=s["game_seed"],
                n_factions=s["n_factions"],
                actor=s["actor"],
                completion_start=s["completion_start"],
                length=s["length"],
            )
            for s in samples
        )
    return refs


def split(refs: list[SampleRef], mod: int, rem: int) -> tuple[list[SampleRef], list[SampleRef]]:
    train = [r for r in refs if r.game_seed % mod != rem]
    val = [r for r in refs if r.game_seed % mod == rem]
    return train, val


def load_sample(ref: SampleRef) -> tuple[torch.Tensor, torch.Tensor]:
    """Full token ids [n] (int64) and target features [n, H] (bf16) of one sample."""
    with safe_open(ref.shard, "pt") as f:
        ids = torch.cat([f.get_tensor(f"t{ref.turn}.prefix_ids"), f.get_tensor(f"s{ref.k}.ids")])
        feats = torch.cat(
            [f.get_tensor(f"t{ref.turn}.prefix_hidden"), f.get_tensor(f"s{ref.k}.hidden")]
        )
    assert len(ids) == len(feats) == ref.length, ref
    return ids.long(), feats


class Samples(torch.utils.data.IterableDataset):
    """Shuffled per epoch; workers take disjoint slices of the same order."""

    def __init__(self, refs: list[SampleRef], seed: int) -> None:
        self.refs = refs
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[SampleRef, torch.Tensor, torch.Tensor]]:
        order = list(range(len(self.refs)))
        random.Random(self.seed + self.epoch).shuffle(order)
        info = torch.utils.data.get_worker_info()
        if info is not None:
            order = order[info.id :: info.num_workers]
        for i in order:
            ref = self.refs[i]
            yield (ref, *load_sample(ref))


def head_inputs(
    head: DraftHead, emb: torch.Tensor, ids: torch.Tensor, feats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Head inputs at positions 0..n-2: (embed(x_{i+1}), f_i). Returns x [1, n-1, H], pos."""
    x = head.inputs(emb[ids[1:]][None], feats[:-1][None])
    return x, torch.arange(len(ids) - 1, device=ids.device)


def sample_loss(
    head: DraftHead,
    emb: torch.Tensor,
    ids: torch.Tensor,
    feats: torch.Tensor,
    completion_start: int,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    noisy = feats + (torch.rand_like(feats) * 2 - 1) * cfg.feature_noise
    x, pos = head_inputs(head, emb, ids, noisy)
    k, v = head.context(x, pos[None])
    a = torch.arange(completion_start - 1, len(ids) - 1, device=ids.device)  # predicts f_{a+1}
    mask = pos[None, :] <= a[:, None]
    out = head.forward_queries(x[:, a], a[None], k, v, mask)[0].float()
    target = feats[a + 1].float()
    reg = F.smooth_l1_loss(out, target)
    with torch.no_grad():
        p_target = torch.softmax((feats[a + 1] @ emb.T).float(), dim=-1)
    logp = torch.log_softmax((out.to(emb.dtype) @ emb.T).float(), dim=-1)
    cls = -(p_target * logp).sum(-1).mean()
    loss = cfg.w_reg * reg + cfg.w_cls * cls
    return loss, {"reg": reg.item(), "cls": cls.item(), "positions": len(a)}


@torch.no_grad()
def chain_accept(
    head: DraftHead,
    emb: torch.Tensor,
    ids: torch.Tensor,
    feats: torch.Tensor,
    completion_start: int,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy chained drafting from every anchor a in [c-1, n-3].

    Returns anchors a and accepted lengths L (0..depth): the number of leading draft steps
    whose token equals the logged token x_{a+1+d}. Draft step d runs at position a+d-1 and
    attends to target-feature keys at positions <= a plus its own earlier draft steps.
    """
    n = len(ids)
    a = torch.arange(completion_start - 1, n - 2, device=ids.device)
    if len(a) == 0:
        return a, torch.zeros(0, dtype=torch.long, device=ids.device)
    x, pos = head_inputs(head, emb, ids, feats)
    k, v = head.context(x, pos[None])
    ctx_mask = pos[None, :] <= a[:, None]
    eye = torch.eye(len(a), dtype=torch.bool, device=ids.device)
    keys, values, masks = [k], [v], [ctx_mask]
    xq, qpos = x[:, a], a
    alive = torch.ones(len(a), dtype=torch.bool, device=ids.device)
    accepted = torch.zeros(len(a), dtype=torch.long, device=ids.device)
    for d in range(1, depth + 1):
        out = head.forward_queries(
            xq, qpos[None], torch.cat(keys, dim=2), torch.cat(values, dim=2), torch.cat(masks, 1)
        )[0]
        tok = (out.to(emb.dtype) @ emb.T).argmax(-1)
        want = a + d + 1
        hit = (want < n) & (tok == ids[want.clamp(max=n - 1)])
        alive &= hit
        accepted += alive.long()
        if d == depth:
            break
        qpos = qpos + 1
        xq = head.inputs(emb[tok][None], out[None].to(x.dtype))
        kd, vd = head.context(xq, qpos[None])
        keys.append(kd)
        values.append(vd)
        masks.append(eye)
    return a, accepted


def token_regions(tokenizer: Any, completion_ids: list[int], actor: str) -> list[str]:
    """Region of each completion token: narrator, or a faction reply's json / message text."""
    if actor == "narrator":
        return ["narrator"] * len(completion_ids)
    starts = [len(tokenizer.decode(completion_ids[:j])) for j in range(len(completion_ids))]
    m = MESSAGE_VALUE.search(tokenizer.decode(completion_ids))
    lo, hi = (m.start(1), m.end(1)) if m else (-1, -1)
    return ["message" if lo <= s < hi else "json" for s in starts]


class AcceptStats:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.n: dict[str, int] = defaultdict(int)
        self.total: dict[str, int] = defaultdict(int)
        self.reached: dict[str, list[int]] = defaultdict(lambda: [0] * depth)

    def add(self, key: str, accepted: list[int]) -> None:
        self.n[key] += len(accepted)
        self.total[key] += sum(accepted)
        for L in accepted:
            for d in range(L):
                self.reached[key][d] += 1

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "anchors": self.n[key],
                # vLLM's "mean acceptance length" counts the bonus token: 1 + accepted drafts.
                "mean_acceptance_length": 1 + self.total[key] / self.n[key],
                "per_position_acceptance": [r / self.n[key] for r in self.reached[key]],
            }
            for key in sorted(self.n)
        }


def evaluate(
    head: DraftHead,
    emb: torch.Tensor,
    tokenizer: Any,
    refs: list[SampleRef],
    depth: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    head.eval()
    stats = AcceptStats(depth)
    for ref in refs:
        ids, feats = load_sample(ref)
        ids, feats = ids.to(device), feats.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            a, acc = chain_accept(head, emb, ids, feats, ref.completion_start, depth)
        c = ref.completion_start
        regions = token_regions(tokenizer, ids[c:].tolist(), ref.actor)
        for anchor, L in zip(a.tolist(), acc.tolist(), strict=True):
            region = regions[anchor + 2 - c]  # region of the first drafted token
            for key in ("all", region, f"n{ref.n_factions}"):
                stats.add(key, [L])
    head.train()
    return stats.summary()


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def load_target_embedding(target: str, device: torch.device) -> torch.Tensor:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(target, dtype=torch.bfloat16)
    emb = model.get_input_embeddings().weight.detach().to(device)
    del model
    return emb.requires_grad_(False)


class Log:
    def __init__(self, path: Path) -> None:
        self.f = path.open("a", encoding="utf-8")

    def __call__(self, row: dict[str, Any]) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        print(json.dumps(row), flush=True)


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoConfig, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-steps", type=int, default=None, help="stop early (smoke runs)")
    parser.add_argument("--eval-samples", type=int, default=None, help="cap the final eval")
    args = parser.parse_args(argv)

    cfg = TrainConfig.model_validate(yaml.safe_load(args.config.read_text("utf-8")))
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / cfg.results_dir / cfg.name / stamp
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), "utf-8")
    run_dir = Path(cfg.runs_dir) / cfg.name / stamp
    run_dir.mkdir(parents=True)
    log = Log(out / "train_log.jsonl")

    refs = index_shards(Path(cfg.hidden_dir))
    train, val = split(refs, cfg.val_seed_mod, cfg.val_seed_rem)
    val_seeds = sorted({(r.n_factions, r.game_seed) for r in val})
    val_sub = random.Random(cfg.seed).sample(val, min(cfg.eval_samples, len(val)))
    log({"event": "data", "train": len(train), "val": len(val), "val_games": val_seeds})

    emb = load_target_embedding(cfg.target, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.target)
    head = DraftHead(HeadConfig.from_target(AutoConfig.from_pretrained(cfg.target).to_dict()))
    head.to(device).train()
    opt = torch.optim.AdamW(
        head.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay
    )
    total = cfg.epochs * len(train) // cfg.accum
    if args.max_steps is not None:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, total, cfg))

    def run_eval(step: int, refs_: list[SampleRef], final: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        res = evaluate(head, emb, tokenizer, refs_, cfg.eval_depth, device)
        row = {"event": "eval", "step": step, "final": final, "samples": len(refs_)}
        log(row | {"seconds": round(time.perf_counter() - t0, 1), "accept": res})
        return res

    run_eval(0, val_sub)
    data = Samples(train, cfg.seed)
    step, t0, acc_stats = 0, time.perf_counter(), defaultdict(float)
    done = False
    for epoch in range(cfg.epochs):
        data.epoch = epoch
        loader = torch.utils.data.DataLoader(
            data, batch_size=None, num_workers=cfg.num_workers, persistent_workers=False
        )
        for i, (ref, ids, feats) in enumerate(loader):
            ids, feats = ids.to(device, non_blocking=True), feats.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss, st = sample_loss(head, emb, ids, feats, ref.completion_start, cfg)
            (loss / cfg.accum).backward()
            for key, value in st.items():
                acc_stats[key] += value
            acc_stats["samples"] += 1
            acc_stats["tokens"] += len(ids)
            if (i + 1) % cfg.accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.log_every_steps == 0:
                dt = time.perf_counter() - t0
                n = acc_stats["samples"]
                log(
                    {
                        "event": "train",
                        "step": step,
                        "epoch": epoch,
                        "lr": sched.get_last_lr()[0],
                        "reg": acc_stats["reg"] / n,
                        "cls": acc_stats["cls"] / n,
                        "grad_norm": grad_norm.item(),
                        "samples_per_s": n / dt,
                        "tokens_per_s": acc_stats["tokens"] / dt,
                    }
                )
                t0, acc_stats = time.perf_counter(), defaultdict(float)
            if step % cfg.eval_every_steps == 0:
                run_eval(step, val_sub)
            if step >= total:
                done = True
                break
        torch.save({"head": head.state_dict(), "step": step}, run_dir / f"epoch{epoch}.pt")
        if done:
            break

    final_refs = val if args.eval_samples is None else val[: args.eval_samples]
    final = run_eval(step, final_refs, final=True)
    head_dir = Path(cfg.heads_dir) / f"{cfg.name}-{stamp}"
    export(head, head_dir)
    summary = {"step": step, "head": str(head_dir), "checkpoints": str(run_dir), "eval": final}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    log({"event": "done", **summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
