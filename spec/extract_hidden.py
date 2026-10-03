"""Phase 2b (trainer container): target hidden states for collected games, one shard per game.

    docker compose --profile train run --rm trainer \
        python -m spec.extract_hidden data/collect/<name>/<run> --out /data/hidden/<name>/<run>

Within a turn every request starts with the same tokens (the shared system prefix is byte
identical across factions and the narrator), and causal attention makes the hidden states of
those positions identical too. So each turn's common prefix is run once, its KV cache is reused
for every sample's suffix (rest of prompt + completion) and cropped back after each, and the
prefix hidden states are stored once per turn. Same mechanism as vLLM's prefix caching; it
cuts both storage and target compute by roughly the number of requests per turn.

Stored hidden states are the target's final-norm output (`last_hidden_state`), which is what
vLLM v0.30 passes to an EAGLE-1 drafter. Shard layout (safetensors, bf16 hidden, int32 ids):

    t<turn>.prefix_ids     [P]      common prefix tokens of that turn
    t<turn>.prefix_hidden  [P, H]
    s<k>.ids               [L_k]    sample k's tokens after the prefix (prompt rest + completion)
    s<k>.hidden            [L_k, H]

Full sample k = cat(t.prefix_*, s<k>.*). Per-sample metadata (turn, actor, completion_start
etc.) is JSON in the shard's `samples` metadata field. This module must not import sim/bench:
the container only has the pinned vLLM image's packages.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

USABLE_FINISH = ("stop", "length")


@dataclass
class Encoded:
    row: dict[str, Any]
    ids: list[int]
    completion_start: int  # index of the first completion token in ids
    prompt_matches: bool | None  # re-tokenized prompt length == vLLM's prompt_tokens
    completion_matches: bool | None  # re-tokenized completion length == completion_tokens


def usable(row: dict[str, Any]) -> bool:
    return (
        row["error"] is None and bool(row["completion"]) and row["finish_reason"] in USABLE_FINISH
    )


def encode(tokenizer: Any, row: dict[str, Any], stop_id: int) -> Encoded:
    """Token ids as the server saw them: chat template + completion (+ stop token if it stopped).

    The completion is re-tokenized from text, which can differ from the sampled ids when the
    model emitted a non-canonical split; the length check against the server's count flags it.
    """
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True
    )
    prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
    completion = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    if row["finish_reason"] == "stop":
        completion = [*completion, stop_id]
    pt, ct = row["prompt_tokens"], row["completion_tokens"]
    return Encoded(
        row=row,
        ids=prompt + completion,
        completion_start=len(prompt),
        prompt_matches=None if pt is None else len(prompt) == pt,
        completion_matches=None if ct is None else len(completion) == ct,
    )


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Longest common prefix, capped so every sequence keeps at least one token of its own."""
    n = min(len(s) for s in seqs) - 1
    first = seqs[0]
    for i in range(n):
        t = first[i]
        if any(s[i] != t for s in seqs[1:]):
            return i
    return max(n, 0)


@torch.inference_mode()
def turn_hidden(
    model: torch.nn.Module, seqs: list[list[int]], prefix_len: int, device: torch.device
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Hidden states for one turn: the shared prefix once, then each suffix on its KV cache."""
    prefix = torch.tensor([seqs[0][:prefix_len]], device=device)
    out = model(input_ids=prefix, use_cache=True)
    cache = out.past_key_values
    prefix_hidden = out.last_hidden_state[0]
    suffixes = []
    for seq in seqs:
        ids = torch.tensor([seq[prefix_len:]], device=device)
        o = model(input_ids=ids, past_key_values=cache, use_cache=True)
        suffixes.append(o.last_hidden_state[0])
        cache.crop(-ids.shape[1])  # drop this suffix; positive crop is deprecated in 5.18
    return prefix_hidden, suffixes


def group_by_turn(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    turns: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        turns[r["turn"]].append(r)
    return dict(sorted(turns.items()))


def extract_game(
    model: torch.nn.Module,
    tokenizer: Any,
    stop_id: int,
    rows: list[dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, int]]:
    tensors: dict[str, torch.Tensor] = {}
    samples: list[dict[str, Any]] = []
    stats = defaultdict(int)
    for turn, turn_rows in group_by_turn(rows).items():
        encoded = [encode(tokenizer, r, stop_id) for r in turn_rows if usable(r)]
        stats["skipped"] += len(turn_rows) - len(encoded)
        if not encoded:
            continue
        seqs = [e.ids for e in encoded]
        p = common_prefix_len(seqs)
        prefix_hidden, suffix_hidden = turn_hidden(model, seqs, p, device)
        tensors[f"t{turn}.prefix_ids"] = torch.tensor(seqs[0][:p], dtype=torch.int32)
        tensors[f"t{turn}.prefix_hidden"] = prefix_hidden.to(dtype).cpu()
        for e, h in zip(encoded, suffix_hidden, strict=True):
            k = len(samples)
            tensors[f"s{k}.ids"] = torch.tensor(e.ids[p:], dtype=torch.int32)
            tensors[f"s{k}.hidden"] = h.to(dtype).cpu()
            r = e.row
            samples.append(
                {
                    "k": k,
                    "turn": turn,
                    "game_seed": r["game_seed"],
                    "n_factions": r["n_factions"],
                    "actor": r["actor"],
                    "faction": r["faction"],
                    "prefix_len": p,
                    "completion_start": e.completion_start,
                    "length": len(e.ids),
                    "finish_reason": r["finish_reason"],
                    "guided": r["guided"],
                    "valid_json": r["valid_json"],
                    "legal": r["legal"],
                    "prompt_matches": e.prompt_matches,
                    "completion_matches": e.completion_matches,
                }
            )
            stats["samples"] += 1
            stats["tokens_total"] += len(e.ids)
            stats["completion_tokens"] += len(e.ids) - e.completion_start
            stats["prompt_mismatch"] += e.prompt_matches is False
            stats["completion_mismatch"] += e.completion_matches is False
        stats["tokens_stored"] += p + sum(len(s) - p for s in seqs)
    return tensors, samples, dict(stats)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_shard(path: Path, tensors: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={k: json.dumps(v) for k, v in meta.items()})
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collect_dir", type=Path, help="a spec.collect run directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None, help="default: model in the collect config")
    parser.add_argument("--stop-token", default="<|im_end|>")
    parser.add_argument("--limit", type=int, default=None, help="only the first N games")
    args = parser.parse_args(argv)

    collect_cfg = yaml.safe_load((args.collect_dir / "config.yaml").read_text("utf-8"))
    model_name = args.model or collect_cfg.get("model")
    if model_name is None:
        parser.error("collect config has no model id; pass --model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stop_id = tokenizer.convert_tokens_to_ids(args.stop_token)
    model = AutoModel.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    games = sorted((args.collect_dir / "games").glob("*.jsonl"))[: args.limit]
    totals: dict[str, int] = defaultdict(int)
    start = time.perf_counter()
    for i, game in enumerate(games):
        shard = args.out / f"{game.stem}.safetensors"
        if shard.exists():
            continue
        tensors, samples, stats = extract_game(
            model, tokenizer, stop_id, load_rows(game), device, dtype
        )
        meta = {"samples": samples, "stats": stats, "model": model_name, "source": game.name}
        write_shard(shard, tensors, meta)
        for k, v in stats.items():
            totals[k] += v
        elapsed = time.perf_counter() - start
        print(f"[{i + 1}/{len(games)}] {game.stem}: {stats} ({elapsed:.0f}s)", flush=True)
    summary = {"model": model_name, "games": len(games), **totals}
    (args.out / "extract_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
