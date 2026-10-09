"""Controlled micro-benchmark: where does a speculative decode step's time go, and does the
trained head transfer to other workloads?

    uv run python -m spec.microbench configs/microbench/<name>.yaml [--server NAME]

For each server config (no speculation, n-gram, random head, trained head ...) one vLLM
container is started with `--cudagraph-metrics` (CUDA graph mode actually used per step).
Then:

- `decode` phases send B real faction prompts from one recorded game turn at once (fixed
  context, shared prefix, as in the game), with and without the JSON grammar. vLLM streams
  one chunk per request per engine step, so the step time is the median gap between
  consecutive chunks while all B requests are decoding (after the last first token, before
  the first finish). vLLM's own per-iteration "elapsed time" cannot be used: with async
  scheduling it times only the scheduler call. Differences between servers isolate
  verification cost (n-gram: extra tokens, almost no draft compute), draft cost (random head:
  always rejected) and grammar cost (guided on vs off).
- `workload` phases send other request mixes (the game turn, a different JSON schema, free
  prose) and read vLLM's speculative acceptance from /metrics deltas.

Writes results/microbench/<name>/<stamp>/<server>/: env.json, vllm.log, phases.jsonl, and a
summary.md for the whole run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics as st
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from bench.client import ChatRequest, Message, OpenAIChatClient
from bench.config import ServerConfig, load_model_config, pinned_image
from bench.metrics import parse_prometheus
from bench.server import VLLMServer, parse_startup_log, vllm_args
from bench.sysinfo import WSL2_CAVEAT, git_info, gpu_info, host_info, image_id
from pydantic import BaseModel, ConfigDict, Field
from sim.actions import response_json_schema

from spec.collect import REPO_ROOT

DIAG_FLAGS = ["--cudagraph-metrics"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerVariant(_Strict):
    name: str
    extra_flags: list[str] = Field(default_factory=list)
    phases: list[str]  # phase names to run against this server


class DecodePhase(_Strict):
    kind: Literal["decode"]
    name: str
    batch_sizes: list[int]
    guided: bool
    max_tokens: int
    # min_tokens = max_tokens + ignore_eos. Only for unguided phases: a closed JSON grammar
    # cannot be forced to continue. Guided phases use natural lengths; the step filter keeps
    # only steps that still carry all B requests.
    fixed_length: bool
    repeats: int


class WorkloadPhase(_Strict):
    kind: Literal["workload"]
    name: str
    source: Literal["game_turn", "ticket_json", "prose"]
    requests: int
    max_tokens: int
    concurrency: int


class MicrobenchConfig(_Strict):
    name: str
    description: str
    model_config_path: str
    game_turn: str  # games/*.jsonl file and turn: "<path>#<turn>"
    max_message_chars: int
    temperature: float
    seed: int
    request_timeout_s: float
    server: ServerConfig
    phases: list[DecodePhase | WorkloadPhase]
    servers: list[ServerVariant]


# --- request sources ----------------------------------------------------------------------


def game_turn_messages(spec: str) -> list[list[Message]]:
    """Faction prompts of one recorded turn, in faction order (shared prefix, ~equal length)."""
    path, turn = spec.rsplit("#", 1)
    rows = [json.loads(line) for line in (REPO_ROOT / path).read_text("utf-8").splitlines()]
    turn_rows = [r for r in rows if r["turn"] == int(turn) and r["actor"] == "faction"]
    return [r["messages"] for r in sorted(turn_rows, key=lambda r: r["faction"])]


TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "customer_name": {"type": "string", "maxLength": 60},
        "product": {"type": "string", "enum": ["router", "laptop", "phone", "printer", "tv"]},
        "issue_category": {
            "type": "string",
            "enum": ["billing", "hardware_fault", "connectivity", "returns", "how_to"],
        },
        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        "order_number": {"type": "string", "maxLength": 20},
        "summary": {"type": "string", "maxLength": 200},
    },
    "required": [
        "customer_name", "product", "issue_category", "priority", "order_number", "summary"
    ],
    "additionalProperties": False,
}  # fmt: skip

_NAMES = ["Ana Ruiz", "Tom Becker", "Li Wei", "Priya Nair", "Omar Haddad", "Sofia Rossi",
          "James Okafor", "Mia Larsen", "Kenji Sato", "Elena Petrova"]  # fmt: skip
_PROBLEMS = {
    "router": ["drops the connection every evening", "lights blink orange and nothing loads"],
    "laptop": ["battery dies after an hour", "screen flickers when I open it past 90 degrees"],
    "phone": ["was charged twice on my card", "speaker crackles during calls"],
    "printer": ["prints blank pages since the update", "I cannot find how to scan to email"],
    "tv": ["arrived with a cracked panel and I want to send it back", "has no sound over HDMI"],
}


def ticket_messages(n: int, rng: random.Random) -> list[list[Message]]:
    """Support tickets to extract into a JSON schema unrelated to the game's."""
    out = []
    for _ in range(n):
        product = rng.choice(list(_PROBLEMS))
        text = (
            f"Hi, this is {rng.choice(_NAMES)}. My {product} (order #{rng.randint(10**6, 10**7)}) "
            f"{rng.choice(_PROBLEMS[product])}. I bought it {rng.randint(2, 40)} days ago and "
            f"this is the {rng.choice(['first', 'second', 'third'])} time I am writing. "
            "Please help."
        )
        out.append(
            [
                {"role": "system", "content": "You extract support tickets into JSON."},
                {"role": "user", "content": f"Ticket:\n{text}\n\nReturn the ticket as JSON."},
            ]
        )
    return out


_TOPICS = ["a lighthouse keeper during a storm", "a market morning in an old port town",
           "two chess players in a park", "a night train crossing mountains",
           "a baker opening shop before dawn", "a lost dog finding its way home",
           "a scientist reading old field notes", "a city after the first snow"]  # fmt: skip


def prose_messages(n: int, rng: random.Random) -> list[list[Message]]:
    return [
        [
            {
                "role": "user",
                "content": f"Write a short descriptive scene about {rng.choice(_TOPICS)}.",
            }
        ]
        for _ in range(n)
    ]


# --- measurement --------------------------------------------------------------------------


def decode_step_stats(chunk_times: list[list[float]]) -> dict[str, Any] | None:
    """Median gap between consecutive streamed chunks while every request is decoding.

    chunk_times[i] holds request i's chunk arrival times. The window starts at the last
    request's first chunk (all prefills done) and ends at the first request's last chunk
    (nobody has finished), so every gap inside it is a step with exactly B requests.
    """
    if not chunk_times or any(len(t) < 2 for t in chunk_times):
        return None
    lo = max(t[0] for t in chunk_times)
    hi = min(t[-1] for t in chunk_times)
    gaps = [b - a for t in chunk_times for a, b in pairwise(t) if lo <= a and b <= hi]
    if len(gaps) < 3:
        return None
    return {"gaps": len(gaps), "step_ms_median": st.median(gaps) * 1000}


SPEC = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted": "vllm:spec_decode_num_accepted_tokens_total",
}


async def spec_counters(root_url: str) -> dict[str, float]:
    async with httpx.AsyncClient() as c:
        text = (await c.get(f"{root_url}/metrics", timeout=10)).text
    samples = parse_prometheus(text)
    return {
        key: sum(v for k, v in samples.items() if k == name or k.startswith(name + "{"))
        for key, name in SPEC.items()
    }


async def run_requests(
    client: OpenAIChatClient,
    messages: list[list[Message]],
    max_tokens: int,
    temperature: float,
    schema: dict[str, Any] | None,
    fixed_length: bool,
    concurrency: int,
    seed: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int, msgs: list[Message]) -> dict[str, Any]:
        times: list[float] = []
        req = ChatRequest(
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed + i,
            json_schema=schema,
            min_tokens=max_tokens if fixed_length else None,
            ignore_eos=fixed_length,
        )
        async with sem:
            c = await client.chat(req, on_token=lambda _: times.append(time.perf_counter()))
        return {
            "tokens": c.completion_tokens,
            "latency_s": c.latency_s,
            "error": c.error,
            "chunk_times": times,
        }

    return list(await asyncio.gather(*(one(i, m) for i, m in enumerate(messages))))


async def run_server_phases(
    cfg: MicrobenchConfig, variant: ServerVariant, server: VLLMServer, out: Path
) -> list[dict[str, Any]]:
    model = load_model_config(cfg.model_config_path)
    client = OpenAIChatClient(cfg.server.base_url, model.model, cfg.request_timeout_s)
    game = game_turn_messages(cfg.game_turn)
    rng = random.Random(cfg.seed)
    phases = {p.name: p for p in cfg.phases}
    rows = []
    try:
        # Warm up graphs/compile caches on the game prefix (not measured).
        await run_requests(client, game[:4], 16, cfg.temperature, None, True, 4, cfg.seed)
        for name in variant.phases:
            p = phases[name]
            if isinstance(p, DecodePhase):
                schema = response_json_schema(cfg.max_message_chars) if p.guided else None
                for b in p.batch_sizes:
                    if b > len(game):
                        raise ValueError(f"batch {b} > {len(game)} prompts in the game turn")
                    for r in range(p.repeats):
                        before = await spec_counters(cfg.server.root_url)
                        t0 = time.perf_counter()
                        res = await run_requests(
                            client, game[:b], p.max_tokens, cfg.temperature, schema,
                            p.fixed_length, b, cfg.seed + 1000 * r,
                        )  # fmt: skip
                        wall = time.perf_counter() - t0
                        after = await spec_counters(cfg.server.root_url)
                        drafts = after["drafts"] - before["drafts"]
                        accepted = after["accepted"] - before["accepted"]
                        rows.append(
                            {
                                "phase": name, "kind": "decode", "guided": p.guided,
                                "batch": b, "repeat": r, "wall_s": wall,
                                "errors": sum(x["error"] is not None for x in res),
                                "decode": decode_step_stats([x["chunk_times"] for x in res]),
                                "output_tokens": sum(x["tokens"] or 0 for x in res),
                                "chunks": sum(len(x["chunk_times"]) for x in res),
                                "mean_acceptance_length": 1 + accepted / drafts if drafts else None,
                            }
                        )  # fmt: skip
            else:
                if p.source == "game_turn":
                    msgs = (game * (p.requests // len(game) + 1))[: p.requests]
                    schema = response_json_schema(cfg.max_message_chars)
                elif p.source == "ticket_json":
                    msgs, schema = ticket_messages(p.requests, rng), TICKET_SCHEMA
                else:
                    msgs, schema = prose_messages(p.requests, rng), None
                before = await spec_counters(cfg.server.root_url)
                res = await run_requests(
                    client, msgs, p.max_tokens, cfg.temperature, schema, False, p.concurrency,
                    cfg.seed,
                )  # fmt: skip
                after = await spec_counters(cfg.server.root_url)
                d = {k: after[k] - before[k] for k in SPEC}
                mal = 1 + d["accepted"] / d["drafts"] if d["drafts"] else None
                rate = d["accepted"] / d["draft_tokens"] if d["draft_tokens"] else None
                rows.append(
                    {
                        "phase": name, "kind": "workload", "source": p.source,
                        "requests": len(msgs), "errors": sum(x["error"] is not None for x in res),
                        "output_tokens": sum(x["tokens"] or 0 for x in res),
                        "mean_acceptance_length": mal,
                        "draft_acceptance_rate": rate,
                        **{f"delta_{k}": v for k, v in d.items()},
                    }
                )  # fmt: skip
            with (out / "phases.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1]) + "\n")
            print(f"  {variant.name} {name}: {json.dumps(rows[-1])[:200]}", flush=True)
    finally:
        await client.aclose()
    return rows


def summarize(results: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["## Decode step time (median over pure-decode steps, ms)", ""]
    decode = [(s, r) for s, rows in results.items() for r in rows if r["kind"] == "decode"]
    keys = sorted({(r["guided"], r["batch"]) for _, r in decode})
    servers = list(results)
    lines.append("| guided | batch | " + " | ".join(servers) + " |")
    lines.append("|---|---:|" + "---:|" * len(servers))
    for guided, b in keys:
        cells = []
        for s in servers:
            vals = [
                r["decode"]["step_ms_median"]
                for srv, r in decode
                if srv == s and r["guided"] == guided and r["batch"] == b and r["decode"]
            ]
            cells.append(f"{st.mean(vals):.1f}" if vals else "-")
        lines.append(f"| {guided} | {b} | " + " | ".join(cells) + " |")
    lines += ["", "## Speculative acceptance by workload", ""]
    lines.append("| server | workload | mean acceptance length | draft acceptance rate |")
    lines.append("|---|---|---:|---:|")
    for s, rows in results.items():
        for r in rows:
            if r["kind"] == "workload" and r["mean_acceptance_length"] is not None:
                lines.append(
                    f"| {s} | {r['source']} | {r['mean_acceptance_length']:.2f} "
                    f"| {r['draft_acceptance_rate']:.2f} |"
                )
    return "\n".join(lines) + f"\n\n{WSL2_CAVEAT}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--server", action="append", help="only these server variants")
    args = parser.parse_args(argv)

    cfg = MicrobenchConfig.model_validate(yaml.safe_load(args.config.read_text("utf-8")))
    model = load_model_config(cfg.model_config_path)
    image = pinned_image()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = REPO_ROOT / "results" / "microbench" / cfg.name / stamp
    results: dict[str, list[dict[str, Any]]] = {}
    for variant in cfg.servers:
        if args.server and variant.name not in args.server:
            continue
        server_cfg = cfg.server.model_copy(
            update={"extra_flags": [*cfg.server.extra_flags, *DIAG_FLAGS, *variant.extra_flags]}
        )
        out = root / variant.name
        out.mkdir(parents=True)
        server = VLLMServer(image, model, server_cfg)
        print(f"[{variant.name}] starting vLLM", flush=True)
        with server:
            env = {
                "started_at": datetime.now(UTC).isoformat(),
                "caveat": WSL2_CAVEAT,
                "vllm_image": image,
                "vllm_image_id": image_id(image),
                "vllm_flags": vllm_args(model, server_cfg),
                "git": git_info(),
                "gpu_at_start": gpu_info(),
                "host": host_info(),
                "kv_actual": parse_startup_log(server.logs()).model_dump(),
            }
            (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
            try:
                results[variant.name] = asyncio.run(run_server_phases(cfg, variant, server, out))
            finally:
                (out / "vllm.log").write_text(server.logs(), encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), "utf-8")
    summary = summarize(results)
    (root / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
