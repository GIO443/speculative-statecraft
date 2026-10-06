import json
from pathlib import Path

from analysis.in_flight import points


def _variant(root: Path, name: str, seconds: float, prompt_tokens: int) -> None:
    d = root / name
    d.mkdir(parents=True)
    summary = {"4": {"seconds_per_turn_mean": seconds}}
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    reqs = [
        {"n_factions": 4, "actor": "faction", "warmup": False, "prompt_tokens": prompt_tokens,
         "ttft_s": 0.1, "latency_s": 0.1 + 0.02 * 9, "output_tokens": 10},
        {"n_factions": 4, "actor": "narrator", "warmup": False, "prompt_tokens": 99999,
         "ttft_s": 0.1, "latency_s": 1.0, "output_tokens": 50},
    ]  # fmt: skip
    (d / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in reqs), encoding="utf-8")


def test_points_use_faction_prompts_and_within_run_ratio(tmp_path: Path) -> None:
    _variant(tmp_path, "baseline", 4.0, 1000)
    _variant(tmp_path, "eagle1-k2", 2.0, 1000)
    (p,) = points(tmp_path, "eagle1-k2")
    assert p["in_flight"] == 4 * 1000  # narrator prompt excluded
    assert p["speedup"] == 2.0
    assert abs(p["tpot_ms"] - 20.0) < 1e-9
