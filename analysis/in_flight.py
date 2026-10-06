"""Put the main sweep and the fixed-prompt control on one axis: total context in flight.

    uv run python -m analysis.in_flight <sweep_dir> <control_dir> --out <png>

Faction count moves concurrency and prompt length together; the control fixes prompt length.
Plotting both against agents x mean prompt tokens per turn shows which variable governs the
speculation speedup and the baseline per-token latency. Only within-run ratios are compared:
raw seconds drift ~6% between sessions.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

BASE = "baseline"


def _requests(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _tpot_ms(reqs: list[dict[str, Any]], n: int) -> float:
    xs = [
        (r["latency_s"] - r["ttft_s"]) / (r["output_tokens"] - 1)
        for r in reqs
        if r["n_factions"] == n
        and r["actor"] == "faction"
        and not r["warmup"]
        and r["ttft_s"]
        and (r["output_tokens"] or 0) > 1
    ]
    return st.median(xs) * 1000


def points(run_dir: Path, variant: str) -> list[dict[str, float]]:
    """Per faction count: tokens in flight, speedup of `variant` over baseline, baseline TPOT."""
    base = json.loads((run_dir / BASE / "summary.json").read_text(encoding="utf-8"))
    spec = json.loads((run_dir / variant / "summary.json").read_text(encoding="utf-8"))
    reqs = _requests(run_dir / BASE / "requests.jsonl")
    out = []
    for key in sorted(base, key=int):
        n = int(key)
        prompts = [
            r["prompt_tokens"]
            for r in reqs
            if r["n_factions"] == n and r["actor"] == "faction" and not r["warmup"]
        ]
        out.append(
            {
                "factions": n,
                "in_flight": n * st.mean(prompts),
                "speedup": base[key]["seconds_per_turn_mean"] / spec[key]["seconds_per_turn_mean"],
                "tpot_ms": _tpot_ms(reqs, n),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--variant", default="eagle1-k2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    series = {
        "prompt grows with agents (main sweep)": points(args.sweep, args.variant),
        "prompt fixed at ~8k tokens (control)": points(args.control, args.variant),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (label, pts), marker in zip(series.items(), ("o", "s"), strict=True):
        x = [p["in_flight"] / 1000 for p in pts]
        ax1.plot(x, [p["speedup"] for p in pts], marker=marker, label=label)
        ax2.plot(x, [p["tpot_ms"] for p in pts], marker=marker, label=label)
        for p, xi in zip(pts, x, strict=True):
            ax1.annotate(f"{p['factions']}", (xi, p["speedup"]), fontsize=7,
                         xytext=(3, 4), textcoords="offset points")  # fmt: skip
    ax1.axhline(1.0, color="grey", lw=1, ls="--")
    ax1.set_ylabel(f"speedup, {args.variant} over no speculation")
    ax2.set_ylabel("baseline time per output token (ms)")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("context in flight: agents x prompt tokens (thousands)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    for label, pts in series.items():
        print(label)
        for p in pts:
            print(f"  {p['factions']:3d} agents  {p['in_flight'] / 1000:6.0f}k in flight  "
                  f"speedup {p['speedup']:.2f}  baseline TPOT {p['tpot_ms']:.1f} ms")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
