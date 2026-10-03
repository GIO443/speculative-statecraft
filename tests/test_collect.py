from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from bench.client import ChatRequest, Completion
from bench.config import REPO_ROOT as SERVING_ROOT
from sim.world import GameConfig, load_game_config

from spec.collect import (
    REPO_ROOT,
    CollectConfig,
    Sample,
    TokenIdClient,
    collect,
    game_path,
    load_collect_config,
    play_game,
)

CONFIGS = REPO_ROOT / "configs" / "collect"
NARRATION = "The realm held its breath."


class FakeClient:
    """Every faction passes; faction 1 replies with invalid JSON; narrator returns a fixed line."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    async def chat(
        self, request: ChatRequest, on_token: Callable[[str], None] | None = None
    ) -> Completion:
        self.requests.append(request)
        last = request.messages[-1]["content"]
        if "chronicler" in last:
            text = NARRATION
        else:
            match = re.search(r"You are faction (\d+)", last)
            assert match is not None
            fid = int(match.group(1))
            body = {"action": {"type": "pass"}, "diplomatic_message": f"faction {fid} waits"}
            text = "not json" if fid == 1 else json.dumps(body)
        ids = request.return_token_ids
        return Completion(
            text=text,
            prompt_tokens=100,
            completion_tokens=len(text),
            ttft_s=0.0,
            latency_s=0.0,
            finish_reason="stop",
            prompt_token_ids=list(range(100)) if ids else None,
            token_ids=[ord(c) for c in text] if ids else None,
        )


@pytest.fixture
def game_cfg() -> GameConfig:
    return load_game_config(SERVING_ROOT / "configs" / "game" / "default.yaml")


@pytest.fixture
def smoke() -> CollectConfig:
    return load_collect_config(CONFIGS / "smoke.yaml")


@pytest.mark.parametrize("name", ["smoke.yaml", "qwen2.5-1.5b.yaml"])
def test_configs_load(name: str) -> None:
    cfg = load_collect_config(CONFIGS / name)
    assert cfg.agent.guided_decoding


def test_reserved_seeds_rejected(tmp_path: Path) -> None:
    text = (
        (CONFIGS / "smoke.yaml")
        .read_text(encoding="utf-8")
        .replace("first_seed: 100", "first_seed: 1")
    )
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="reserved"):
        load_collect_config(path)


def test_play_game_records_exact_requests(smoke: CollectConfig, game_cfg: GameConfig) -> None:
    client = FakeClient()
    samples = asyncio.run(play_game(client, game_cfg, smoke.agent, 4, 100, turns=2))

    # One sample per request, in request order, carrying the exact messages sent.
    assert len(samples) == len(client.requests) == 2 * (4 + 1)
    assert [s.messages for s in samples] == [r.messages for r in client.requests]
    assert [s.turn for s in samples] == [0] * 5 + [1] * 5
    narr = [s for s in samples if s.actor == "narrator"]
    assert [s.completion for s in narr] == [NARRATION, NARRATION]
    assert all(s.valid_json is None and s.legal is None and not s.guided for s in narr)

    factions = [s for s in samples if s.actor == "faction"]
    assert all(s.guided for s in factions)
    bad = [s for s in factions if s.faction == 1]
    assert all(s.valid_json is False and s.legal is False for s in bad)
    good = [s for s in factions if s.faction != 1]
    assert all(s.valid_json and s.legal for s in good)


def test_collect_writes_games_and_resumes(
    smoke: CollectConfig, game_cfg: GameConfig, tmp_path: Path
) -> None:
    totals = asyncio.run(collect(smoke, game_cfg, FakeClient(), tmp_path))
    assert totals == {"games": 1, "skipped": 0, "samples": 10, "errors": 0}
    path = game_path(tmp_path, 4, 100)
    rows = [Sample.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]
    assert len(rows) == 10 and {r.game_seed for r in rows} == {100}
    assert not list(tmp_path.rglob("*.tmp"))

    client = FakeClient()
    again = asyncio.run(collect(smoke, game_cfg, client, tmp_path))
    assert again["skipped"] == 1 and again["games"] == 0
    assert client.requests == []


def test_token_id_client_requests_and_records_ids(
    smoke: CollectConfig, game_cfg: GameConfig
) -> None:
    fake = FakeClient()
    samples = asyncio.run(play_game(TokenIdClient(fake), game_cfg, smoke.agent, 4, 100, turns=1))
    assert fake.requests and all(r.return_token_ids for r in fake.requests)
    for s in samples:
        assert s.prompt_token_ids == list(range(100))
        assert s.token_ids == [ord(c) for c in s.completion]


def test_ids_absent_without_token_id_client(smoke: CollectConfig, game_cfg: GameConfig) -> None:
    samples = asyncio.run(play_game(FakeClient(), game_cfg, smoke.agent, 4, 100, turns=1))
    assert all(s.prompt_token_ids is None and s.token_ids is None for s in samples)
