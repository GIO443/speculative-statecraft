import random
from pathlib import Path

import pytest
import yaml

from spec.collect import REPO_ROOT
from spec.microbench import (
    TICKET_SCHEMA,
    MicrobenchConfig,
    decode_step_stats,
    game_turn_messages,
    prose_messages,
    summarize,
    ticket_messages,
)

CONFIG = REPO_ROOT / "configs" / "microbench" / "step-cost.yaml"
CONFIGS = sorted((REPO_ROOT / "configs" / "microbench").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path: Path) -> None:
    cfg = MicrobenchConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
    phase_names = {p.name for p in cfg.phases}
    assert all(set(s.phases) <= phase_names for s in cfg.servers)
    # A closed JSON grammar cannot be forced to keep going.
    assert all(not (p.guided and p.fixed_length) for p in cfg.phases if p.kind == "decode")


def test_step_time_from_chunk_gaps() -> None:
    # Request A streams every 10 ms from t=0; B's first chunk arrives at t=0.05 (later prefill)
    # and it finishes first at t=0.15. Only gaps inside [0.05, 0.15] count: all 10 ms.
    a = [i * 0.01 for i in range(31)]
    b = [0.05 + i * 0.01 for i in range(11)]
    stats = decode_step_stats([a, b])
    assert stats is not None and abs(stats["step_ms_median"] - 10.0) < 1e-6
    assert stats["gaps"] == 20
    assert decode_step_stats([[0.0], a]) is None


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
    assert len(msgs) == 32
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
