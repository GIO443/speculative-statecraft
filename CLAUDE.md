# CLAUDE.md

## Project overview

Phases 2-4 of the statecraft project. Phase 1 (simulation, serving baseline, benchmark harness)
lives in the sibling repo **statecraft-serving** (`../statecraft-serving`, github.com/GIO443/statecraft-serving),
installed here as an **editable path dependency** for `sim/` and `bench/`. Read its CLAUDE.md and
README for the simulation design, harness usage and Phase 1 findings.

Goal: train our own EAGLE-style draft head on the simulation's own outputs, show it beats generic
speculation on this workload, and find the faction count where speculation stops helping (GPU
becomes compute saturated) and explain why. Every result must be explained by mechanism
(memory-bandwidth-bound decode, compute-bound prefill, KV cache capacity), not just reported.

## Hardware and environment

- Windows host, RTX 4070 Laptop, **8 GB VRAM** (Ada, FP8 native). Shell is **PowerShell**.
- GPU work runs in Docker Desktop (WSL2). vLLM is pinned to **v0.30.0** by digest in
  statecraft-serving's `docker-compose.yml`; reuse that pin. Never benchmark on `latest`.
- Native side: uv-managed Python 3.12, talks to vLLM at `http://localhost:8000/v1`.
- Weights live in the external Docker volume `hf-cache` at `/root/.cache/huggingface`.
- Default `--gpu-memory-utilization` 0.8 (free VRAM is ~6.55 GiB with Windows using ~1.4 GiB).

### Hard constraints

- **Serving and training never run at the same time.** Both need the whole GPU.
- Ask before downloading any model over ~2 GB.
- Benchmark numbers carry a "WSL2 / Docker on Windows" caveat; record it in every results file.
- statecraft-serving must stay an editable install (`bench.config.REPO_ROOT` resolves `configs/`
  and `docker-compose.yml` from that checkout). If its folder moves, rebuild this venv.

## Current status

- Scaffolded 2026-10-02: pyproject with editable path dep, `spec/` package, dependency test.
- Target model: **Qwen2.5-1.5B-Instruct bf16** (model matrix dropped in Phase 1; no downloads needed).
- `spec/collect.py` done: `uv run python -m spec.collect configs/collect/<name>.yaml [--out DIR]
  [--dry-run]` launches the pinned vLLM container and writes
  `data/collect/<name>/<UTC stamp>/games/n<N>_s<seed>.jsonl` (exact messages + completion per
  request, faction and narrator), `config.yaml`, `env-<stamp>.json`, `vllm-<stamp>.log`. Finished
  games are skipped on `--out` resume. Collection seeds start at 100; seeds 0-2 and 1000 are
  reserved for Phase 4 eval and rejected. Settings match Phase 1 `default` (guided on, T=0.7).
  Live smoke run OK (2026-10-03).
- Finding: the narrator sometimes copies an earlier narration **verbatim** from "Recent history"
  in its prompt (smoke: turn 1 = turn 0; Phase 1 default: 1 of 90). Different seeds, so it is
  in-context copying, not a seeding bug. Expect n-gram (prompt lookup) speculation to do very well
  on these; measure the copy rate on the full collection.
- Finding: guided JSON whitespace varies between replies (compact, 2-space, 4-space indent);
  the grammar allows it, so it is real entropy for the draft head.
- Next: full collection (`configs/collect/qwen2.5-1.5b.yaml`, 150 games, ~2 h GPU), then
  `spec/extract_hidden.py`.

## Layout (target)

```
spec/
  collect.py         # run games, log prompts + target completions to JSONL
  extract_hidden.py  # (trainer container) target hidden states -> safetensors shards
  draft_head.py      # EAGLE-style draft head
  train.py           # (trainer container) training loop
  export.py          # export head in a format vLLM v0.30 can load as an EAGLE drafter
docker/trainer.Dockerfile  # FROM the pinned vLLM image; adds training deps
configs/             # spec experiment configs (YAML)
analysis/
results/
tests/
```

## Phases

### Phase 2: Training data collection
- Run many games across seeds and faction counts; log every prompt and the target's completion.
- In the trainer container, extract target hidden states with transformers; write sharded
  safetensors to a shared volume. Stream to disk; never hold the dataset in RAM.

### Phase 3: Draft head
- Implement a minimal EAGLE-style head ourselves (one transformer layer predicting next tokens
  from target hidden states plus token embeddings).
- Hold out games by seed for validation; track acceptance-style metrics offline.
- **Before building the exporter, check vLLM v0.30's source for the EAGLE drafter format and its
  compatibility with guided decoding and prefix caching.** Document incompatibilities as findings.

### Phase 4: Analysis
- Compare: no speculation, n-gram (prompt lookup), a public generic drafter if one exists, ours.
- Mean accepted length by output region: JSON scaffolding, `diplomatic_message`, narrator text.
- Speedup vs faction count; locate and explain the crossover (spare compute at low batch vs
  saturation at high batch). Stretch: enable speculation dynamically based on load.

## Measurement

Same results layout as statecraft-serving (config.yaml, env.json, requests.jsonl, turns.jsonl,
server_metrics.jsonl), plus spec decode acceptance metrics. Run each config at least 3 times,
report mean and spread, warm up before timing.

## Working conventions

- Small, reviewable steps; propose a plan before large changes.
- Python 3.12, type hints, pydantic, `ruff`, `pytest`. Write and run tests + ruff for every change.
- All experiment parameters from YAML; no magic numbers. Never silently change a benchmark
  setting between compared runs; record changes in `env.json`.
- When unsure whether a vLLM flag/feature exists in v0.30, check its docs or source.
