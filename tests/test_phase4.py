import json
from pathlib import Path

from analysis.phase4 import acceptance, rows


def _row(turn: int, n: int, drafts: float, accepted: float, pos: list[float], phase="measure"):
    m = {
        'vllm:spec_decode_num_drafts_total{engine="0"}': drafts,
        'vllm:spec_decode_num_accepted_tokens_total{engine="0"}': accepted,
    }
    for i, v in enumerate(pos):
        m[f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{i}"}}'] = v
    return {"kind": "turn_end", "phase": phase, "n_factions": n, "turn": turn, "metrics": m}


def test_acceptance_counts_measured_turns_only(tmp_path: Path) -> None:
    rows_ = [
        {"kind": "turn_end", "phase": "server_warmup", "turn": 0, "metrics": {}},
        _row(0, 4, 10, 5, [4, 1]),  # warmup turn of the game: excluded
        _row(1, 4, 30, 25, [16, 9]),  # +20 drafts, +20 accepted
        _row(0, 8, 40, 30, [20, 10]),  # next game's warmup turn: excluded
        _row(1, 8, 140, 80, [60, 20]),  # +100 drafts, +50 accepted
    ]
    path = tmp_path / "server_metrics.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows_), encoding="utf-8")
    acc = acceptance(path, warmup_turns=1)
    assert acc[4]["drafts"] == 20 and acc[4]["mean_acceptance_length"] == 2.0
    assert acc[4]["per_position"] == [0.6, 0.4]
    assert acc[8]["mean_acceptance_length"] == 1.5
    assert acc[8]["per_position"] == [0.4, 0.1]


def test_speedup_relative_to_baseline() -> None:
    def v(name: str, s: float) -> dict:
        summary = {4: {"seconds_per_turn_mean": s, "seconds_per_turn_stdev": 0.1,
                       "legal_rate": 0.6, "excluded_turns": 0}}  # fmt: skip
        kv = {"kv_cache_gib": 1.5, "kv_cache_tokens": 1000}
        return {"name": name, "summary": summary, "kv": kv, "acceptance": {}}

    table = rows([v("baseline", 4.0), v("eagle1-k3", 2.0)])
    assert [r["speedup"] for r in table] == [1.0, 2.0]
