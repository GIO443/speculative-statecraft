"""Phase 2 data collection: play seeded games and log every prompt with the target's completion.

    uv run python -m spec.collect configs/collect/<name>.yaml [--out DIR] [--dry-run]

Each game is written to games/n<factions>_s<seed>.jsonl, one row per request (faction decisions
and narrator), holding the exact chat messages sent and the completion text. A game file appears
only once the game finishes, so rerunning with --out on an interrupted run skips finished games.
Token ids are not logged: extract_hidden.py applies the chat template and tokenizes, and checks
the completion token count against the server's `completion_tokens`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from bench.client import ChatClient, Completion, Message, OpenAIChatClient
from bench.config import REPO_ROOT as SERVING_ROOT
from bench.config import ModelConfig, ServerConfig, load_model_config, pinned_image
from bench.server import VLLMServer, parse_startup_log, vllm_args
from bench.sysinfo import WSL2_CAVEAT, git_info, gpu_info, host_info, image_id
from pydantic import BaseModel, ConfigDict, Field
from sim.agents import AgentConfig, Agents, Decision
from sim.engine import Game
from sim.world import GameConfig, load_game_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GameSet(_Strict):
    n_factions: int = Field(ge=2)
    first_seed: int
    n_games: int = Field(ge=1)

    @property
    def seeds(self) -> range:
        return range(self.first_seed, self.first_seed + self.n_games)


class CollectRun(_Strict):
    turns: int = Field(ge=1)
    request_timeout_s: float
    games: list[GameSet] = Field(min_length=1)
    # Seeds the Phase 4 benchmark measures on; collecting on them would leak eval data.
    reserved_seeds: list[int]


class CollectConfig(_Strict):
    name: str
    description: str
    model_config_path: str  # relative to the statecraft-serving checkout
    game_config_path: str  # relative to the statecraft-serving checkout
    server: ServerConfig
    agent: AgentConfig
    run: CollectRun


class Sample(BaseModel):
    """One row of games/*.jsonl: a single request and the target's completion."""

    game_seed: int
    n_factions: int
    turn: int
    actor: Literal["faction", "narrator"]
    faction: int | None
    messages: list[Message]
    completion: str
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    guided: bool  # completion was produced under the JSON-schema grammar
    valid_json: bool | None  # None for narrator
    legal: bool | None  # None for narrator; set once the turn resolves
    error: str | None


def _resolve_serving_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else SERVING_ROOT / p


def load_collect_config(path: Path) -> CollectConfig:
    with path.open(encoding="utf-8") as f:
        cfg = CollectConfig.model_validate(yaml.safe_load(f))
    leaked = {s for g in cfg.run.games for s in g.seeds} & set(cfg.run.reserved_seeds)
    if leaked:
        raise ValueError(f"collection seeds overlap reserved eval seeds: {sorted(leaked)}")
    if cfg.agent.narrator == "pipelined":
        raise ValueError("use narrator: sequential or off; pipelined narration lags history")
    return cfg


class RecordingAgents(Agents):
    """Agents that also keep the exact messages and completion of every request."""

    def __init__(
        self, client: ChatClient, game_cfg: GameConfig, agent_cfg: AgentConfig, seed: int, n: int
    ) -> None:
        super().__init__(client, game_cfg, agent_cfg)
        self.seed = seed
        self.n_factions = n
        self.turn = 0
        self.pending: list[Sample] = []

    def _sample(
        self,
        actor: Literal["faction", "narrator"],
        fid: int | None,
        messages: list[Message],
        c: Completion,
        valid_json: bool | None,
        guided: bool,
    ) -> Sample:
        return Sample(
            game_seed=self.seed,
            n_factions=self.n_factions,
            turn=self.turn,
            actor=actor,
            faction=fid,
            messages=messages,
            completion=c.text,
            prompt_tokens=c.prompt_tokens,
            completion_tokens=c.completion_tokens,
            finish_reason=c.finish_reason,
            guided=guided,
            valid_json=valid_json,
            legal=None,
            error=c.error,
        )

    async def decide(self, fid: int, messages: list[Message], seed: int) -> Decision:
        d = await super().decide(fid, messages, seed)
        guided = self._schema is not None
        valid = d.response is not None
        self.pending.append(self._sample("faction", fid, messages, d.completion, valid, guided))
        return d

    async def narrate(
        self, messages: list[Message], seed: int, on_token: Callable[[str], None] | None
    ) -> Completion:
        c = await super().narrate(messages, seed, on_token)
        self.pending.append(self._sample("narrator", None, messages, c, None, False))
        return c


async def play_game(
    client: ChatClient,
    game_cfg: GameConfig,
    agent_cfg: AgentConfig,
    n_factions: int,
    seed: int,
    turns: int,
) -> list[Sample]:
    game = Game(n_factions, seed, game_cfg, agent_cfg, client)
    agents = RecordingAgents(client, game_cfg, agent_cfg, seed, n_factions)
    game.agents = agents
    samples: list[Sample] = []
    for _ in range(turns):
        if game.over:
            break
        agents.turn = game.world.turn
        rec = await game.play_turn()
        for s in agents.pending:
            if s.actor == "faction":
                s.legal = s.faction in rec.result.legal
        samples.extend(agents.pending)
        agents.pending = []
    return samples


def game_path(out: Path, n_factions: int, seed: int) -> Path:
    return out / "games" / f"n{n_factions}_s{seed}.jsonl"


def write_game(path: Path, samples: list[Sample]) -> None:
    """Write via a temp file so a partial game never looks finished."""
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
    tmp.replace(path)


async def collect(
    cfg: CollectConfig, game_cfg: GameConfig, client: ChatClient, out: Path
) -> dict[str, int]:
    (out / "games").mkdir(parents=True, exist_ok=True)
    totals = {"games": 0, "skipped": 0, "samples": 0, "errors": 0}
    for gs in cfg.run.games:
        for seed in gs.seeds:
            path = game_path(out, gs.n_factions, seed)
            if path.exists():
                totals["skipped"] += 1
                continue
            samples = await play_game(
                client, game_cfg, cfg.agent, gs.n_factions, seed, cfg.run.turns
            )
            write_game(path, samples)
            errors = sum(s.error is not None for s in samples)
            totals["games"] += 1
            totals["samples"] += len(samples)
            totals["errors"] += errors
            print(f"  n={gs.n_factions:>3} seed={seed}: {len(samples)} samples, {errors} errors")
    return totals


def _own_git() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT)
        return r.stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD") or "no commits",
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="resume into this run directory")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    args = parser.parse_args(argv)

    cfg = load_collect_config(args.config)
    model: ModelConfig = load_model_config(cfg.model_config_path)
    game_cfg = load_game_config(_resolve_serving_path(cfg.game_config_path))
    image = pinned_image()
    server = VLLMServer(image, model, cfg.server)
    n_games = sum(g.n_games for g in cfg.run.games)
    print(f"[{cfg.name}] {n_games} games x {cfg.run.turns} turns on {model.model}")
    if args.dry_run:
        print("  " + " ".join(server.command))
        return 0

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or REPO_ROOT / "data" / "collect" / cfg.name / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": model.model,
                "collect": cfg.model_dump(mode="json"),
                "game": game_cfg.model_dump(mode="json"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    with server:
        env = {
            "started_at": datetime.now(UTC).isoformat(),
            "caveat": WSL2_CAVEAT,
            "vllm_image": image,
            "vllm_image_id": image_id(image),
            "vllm_version": server.vllm_version(),
            "model": model.model,
            "quantization_flags": model.served_flags,
            "vllm_flags": vllm_args(model, cfg.server),
            "docker_command": server.command,
            "git": {"speculative-statecraft": _own_git(), "statecraft-serving": git_info()},
            "gpu_at_start": gpu_info(),
            "host": host_info(),
            "kv_actual": parse_startup_log(server.logs()).model_dump(),
        }
        # One env file per session, so a resumed run keeps the record of each server it used.
        (out / f"env-{stamp}.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        client = OpenAIChatClient(cfg.server.base_url, model.model, cfg.run.request_timeout_s)

        async def run() -> dict[str, int]:
            try:
                return await collect(cfg, game_cfg, client, out)
            finally:
                await client.aclose()

        try:
            totals = asyncio.run(run())
        finally:
            (out / f"vllm-{stamp}.log").write_text(server.logs(), encoding="utf-8")
    print(f"done: {totals} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
