"""Phase 4 tables and plots from a bench.harness run directory.

    uv run python -m analysis.phase4 results/phase4-1.5b/<stamp>

Per variant and faction count: seconds per turn (mean, stdev over repeats), speedup over
`baseline`, vLLM's speculative acceptance (mean acceptance length = 1 + accepted / drafts, and
per-position rates) from counter deltas over measured (non-warmup) turns, and the KV cache
vLLM allocated. Writes phase4.md, phase4.csv, seconds_per_turn.png and speedup.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


def _sum(metrics: dict[str, float], name: str) -> float:
    return sum(v for k, v in metrics.items() if k == name or k.startswith(name + "{"))


def _per_pos(metrics: dict[str, float]) -> dict[int, float]:
    out: dict[int, float] = defaultdict(float)
    for k, v in metrics.items():
        if k.startswith(PER_POS + "{"):
            out[int(k.split('position="')[1].split('"')[0])] += v
    return out


def acceptance(metrics_path: Path, warmup_turns: int) -> dict[int, dict[str, Any]]:
    """Per faction count: drafts, accepted, MAL and per-position rates over measured turns."""
    acc: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"drafts": 0.0, "accepted": 0.0, "pos": defaultdict(float)}
    )
    prev: dict[str, float] | None = None
    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") != "turn_end":
                continue
            m = row["metrics"]
            if prev is not None and row["phase"] == "measure" and row["turn"] >= warmup_turns:
                a = acc[row["n_factions"]]
                a["drafts"] += _sum(m, DRAFTS) - _sum(prev, DRAFTS)
                a["accepted"] += _sum(m, ACCEPTED) - _sum(prev, ACCEPTED)
                before = _per_pos(prev)
                for pos, v in _per_pos(m).items():
                    a["pos"][pos] += v - before.get(pos, 0.0)
            prev = m
    out = {}
    for n, a in acc.items():
        if a["drafts"] > 0:
            out[n] = {
                "drafts": int(a["drafts"]),
                "mean_acceptance_length": 1 + a["accepted"] / a["drafts"],
                "per_position": [a["pos"][p] / a["drafts"] for p in sorted(a["pos"])],
            }
    return out


def load_variant(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    env = json.loads((path / "env.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    return {
        "name": path.name,
        "summary": {int(n): s for n, s in summary.items()},
        "kv": env["kv_actual"],
        "acceptance": acceptance(path / "server_metrics.jsonl", cfg["run"]["warmup_turns"]),
    }


def rows(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = next((v for v in variants if v["name"] == "baseline"), None)
    out = []
    for v in variants:
        for n, s in sorted(v["summary"].items()):
            mean = s["seconds_per_turn_mean"]
            base_mean = base["summary"].get(n, {}).get("seconds_per_turn_mean") if base else None
            acc = v["acceptance"].get(n)
            out.append(
                {
                    "variant": v["name"],
                    "factions": n,
                    "s_per_turn": mean,
                    "s_per_turn_sd": s["seconds_per_turn_stdev"],
                    "speedup": base_mean / mean if base_mean and mean else None,
                    "mal": acc["mean_acceptance_length"] if acc else None,
                    "per_pos": acc["per_position"] if acc else None,
                    "kv_gib": v["kv"]["kv_cache_gib"],
                    "kv_tokens": v["kv"]["kv_cache_tokens"],
                    "legal": s["legal_rate"],
                    "excluded": s["excluded_turns"],
                }
            )
    return out


def _fmt(x: Any, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def markdown(table: list[dict[str, Any]], caveat: str) -> str:
    lines = [
        "| variant | factions | s/turn | speedup | MAL | per-position | KV GiB | excluded |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in table:
        pp = "-" if r["per_pos"] is None else " / ".join(f"{p:.2f}" for p in r["per_pos"])
        lines.append(
            f"| {r['variant']} | {r['factions']} | {r['s_per_turn']:.2f} ± "
            f"{r['s_per_turn_sd']:.2f} | {_fmt(r['speedup'], '.2f')} | {_fmt(r['mal'], '.2f')} "
            f"| {pp} | {r['kv_gib']} | {r['excluded']} |"
        )
    return "\n".join(lines) + f"\n\n{caveat}\n"


def plot(table: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in table:
        by[r["variant"]].append(r)
    for key, fname, ylabel in (
        ("s_per_turn", "seconds_per_turn.png", "seconds per world turn"),
        ("speedup", "speedup.png", "speedup over no speculation"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, rs in by.items():
            pts = [(r["factions"], r[key]) for r in rs if r[key] is not None]
            if key == "speedup" and name == "baseline":
                continue
            ax.plot(*zip(*pts, strict=True), marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["factions"] for r in table}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if key == "speedup":
            ax.axhline(1.0, color="grey", lw=1, ls="--")
        else:
            ax.set_yscale("log")
        ax.set_xlabel("factions (concurrent requests per turn)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    variants = [
        load_variant(p)
        for p in sorted(args.run_dir.iterdir())
        if p.is_dir() and (p / "summary.json").exists()
    ]
    caveat = json.loads((args.run_dir / variants[0]["name"] / "env.json").read_text("utf-8"))[
        "caveat"
    ]
    table = rows(variants)
    (args.run_dir / "phase4.md").write_text(markdown(table, caveat), encoding="utf-8")
    with (args.run_dir / "phase4.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    plot(table, args.run_dir)
    print(markdown(table, caveat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
