import random

import pytest
import yaml

from spec.collect import REPO_ROOT
from spec.microbench import (
    TICKET_SCHEMA,
    MicrobenchConfig,
    decode_step_stats,
    game_turn_messages,
    parse_steps,
    prose_messages,
    summarize,
    ticket_messages,
)

CONFIG = REPO_ROOT / "configs" / "microbench" / "step-cost.yaml"


def _line(i: int, ctx_req: int, ctx_tok: int, gen_req: int, gen_tok: int, ms: float) -> str:
    return (
        f"(EngineCore pid=1) INFO 10-07 10:00:00 [loggers.py:190] Iteration({i}): "
        f"{ctx_req} context requests, {ctx_tok} context tokens, {gen_req} generation requests, "
        f"{gen_tok} generation tokens, iteration elapsed time: {ms:.2f} ms, "
        "GPU KV cache usage: 3.0%"
    )


LOG = "\n".join(
    [
        _line(7, 2, 900, 0, 0, 40.1),
        _line(8, 0, 0, 2, 4, 12.5),
        _line(9, 0, 0, 2, 4, 13.5),
        _line(10, 0, 0, 1, 2, 9.0),
    ]
)


def test_config_loads() -> None:
    cfg = MicrobenchConfig.model_validate(yaml.safe_load(CONFIG.read_text("utf-8")))
    phase_names = {p.name for p in cfg.phases}
    assert all(set(s.phases) <= phase_names for s in cfg.servers)
    # A closed JSON grammar cannot be forced to keep going.
    assert all(not (p.guided and p.fixed_length) for p in cfg.phases if p.kind == "decode")


def test_parse_steps_and_pure_decode_filter() -> None:
    steps = parse_steps(LOG)
    assert [s["iteration"] for s in steps] == [7, 8, 9, 10]
    stats = decode_step_stats(steps, 2)
    assert stats == {"steps": 2, "step_ms_median": 13.0, "gen_tokens_per_step": 4.0}
    assert decode_step_stats(steps, 64) is None


def test_request_sources() -> None:
    rng = random.Random(0)
    tickets = ticket_messages(5, rng)
    assert len(tickets) == 5 and all(m[-1]["role"] == "user" for m in tickets)
    assert set(TICKET_SCHEMA["required"]) == set(TICKET_SCHEMA["properties"])
    assert len(prose_messages(3, rng)) == 3


def test_game_turn_messages_share_prefix() -> None:
    cfg = MicrobenchConfig.model_validate(yaml.safe_load(CONFIG.read_text("utf-8")))
    path = REPO_ROOT / cfg.game_turn.rsplit("#", 1)[0]
    if not path.exists():
        pytest.skip("collected data not present")
    msgs = game_turn_messages(cfg.game_turn)
    assert len(msgs) == 64
    assert len({m[0]["content"] for m in msgs}) == 1  # identical system (shared) section


def test_summarize_tables() -> None:
    results = {
        "none": [{"kind": "decode", "guided": True, "batch": 4,
                  "decode": {"step_ms_median": 10.0}}],
        "eagle-k3": [{"kind": "workload", "source": "prose", "mean_acceptance_length": 1.5,
                      "draft_acceptance_rate": 0.2}],
    }  # fmt: skip
    md = summarize(results)
    assert "| True | 4 | 10.0 | - |" in md
    assert "| eagle-k3 | prose | 1.50 | 0.20 |" in md
