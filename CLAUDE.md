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
- vLLM v0.30 source check (2026-10-03, read from the pinned image):
  - No Qwen2-specific EAGLE class. Registered generic drafters: `EagleLlamaForCausalLM`
    (EAGLE-1: `fc(cat(embed, hidden)) -> Llama decoder layers`, first layer's input_layernorm
    skipped) and `Eagle3LlamaForCausalLM` / `LlamaForCausalLMEagle3` (EAGLE-3, fc over
    concatenated aux layers). `EAGLEConfig` names the drafter `Eagle<arch>` from the draft
    config's `architectures`, so our head ships as a Llama-architecture checkpoint.
  - Qwen2 implements `SupportsEagle3` (aux hidden states), so EAGLE-3 is also an option.
  - Draft without its own `embed_tokens` / `lm_head` weights shares the target's (saves VRAM).
  - Structured outputs + spec decode: `StructuredOutputManager.validate_tokens` trims draft
    tokens to the grammar-valid prefix, so guided decoding and speculation coexist.
  - `extract_hidden_states` spec method only writes hidden states into KV cache for KV-transfer
    connectors; not a practical dump path. Extract with transformers in the trainer container.
- **Decisions (2026-10-03):** EAGLE-1 head first (EAGLE-3 maybe later as a comparison).
  Hidden states are stored with per-turn **shared-prefix dedupe**: full sequences are ~80M
  tokens (~240 GB at 3 KiB/token), too big for disk; the common prefix of a turn is stored once.
- Full collection running/ran into `data/collect/qwen2.5-1.5b/run1` (resume with
  `--out data/collect/qwen2.5-1.5b/run1`; log in `data/collect/qwen2.5-1.5b-run1.log`).
  That run's config.yaml predates the `model` field, so pass `--model` to extract_hidden.
- `spec/extract_hidden.py` + trainer container written and tested on CPU (prefix reuse matches
  a full forward). Trainer: `docker compose --profile train run --rm trainer ...` (this repo's
  compose; image pin tested equal to statecraft-serving's). Shards go to the `spec-data` Docker
  volume at `/data`, not the OneDrive-synced checkout. Container tests:
  `docker run --rm -v "${PWD}:/work" -w /work speculative-statecraft-trainer python3 -m pytest tests/container`.
- Collection run1 done (2026-10-03): 150 games, 19,200 faction + 1,200 narrator samples,
  0 errors, 188 MB JSONL; 88.3M prompt + 1.17M completion tokens. Faction valid JSON 100%,
  legal 65.8%. Narrator hit its 200-token cap in 40% (482/1200). Narrator verbatim copies of an
  earlier narration in its prompt: 31/1200 (2.6%).
- extract_hidden GPU check (2 x 16-faction games): prompt token counts match vLLM exactly
  (0/272 mismatches); dedupe stores 127k of 766k tokens (6x), ~195 MB per 16-faction game,
  ~30 GB estimated for run1; ~11 s per 16-faction game.
- Finding: **guided decoding produces non-canonical token splits.** Wherever a faction reply
  contains `"},"` (end of the action object, next key), the tokenizer's canonical encoding is one
  token but the target under the grammar emitted two (90/800 faction replies, always -1 token).
  Narrator +1 mismatches are mostly length-capped replies. Re-tokenizing completion text is
  therefore not exactly what the target sampled. vLLM v0.30 supports `return_token_ids` on chat
  completions (incl. streaming), which would give the sampled ids directly.
- Decision (2026-10-03): re-collect with sampled token ids. statecraft-serving's client got an
  opt-in `return_token_ids` (commit 61f241e, off by default); collect.py turns it on, rows carry
  `prompt_token_ids` / `token_ids` (verified live: counts equal vLLM usage, completions end with
  `<|im_end|>` 151645), and extract_hidden uses them. **run2** (`data/collect/qwen2.5-1.5b/run2`)
  supersedes run1 for training; run1 is kept only as the record of the tokenization finding.
- run2 done: 150 games, 20,400 samples, 0 errors, ids on every row and equal to vLLM usage
  counts. Legal 64.9%, narrator capped 473/1200, narrator verbatim copies 26/1200.
- run2 hidden states extracted to `/data/hidden/qwen2.5-1.5b/run2` (spec-data volume): 150
  shards, 32 GB, ~40 min GPU. Dedupe stores 11.1M of 89.6M sequence tokens (8.0x). 3,075/20,400
  samples (15%) contain at least one non-canonical split vs re-tokenized text (stored as sampled).
- Draft head plan approved 2026-10-03. `spec/draft_head.py` (EAGLE-1, 51.5M params, matches
  vLLM's `EagleLlamaForCausalLM`: `fc(cat(embed, feature))`, layer 0 without input norm, no
  final norm, target's tied embedding as LM head) and `spec/export.py` (config.json +
  model.safetensors with `fc.*` / `layers.0.*` only). CPU tests: layer equals HF
  `LlamaDecoderLayer` with Identity input norm; RoPE equals HF's.
- Export smoke (`configs/experiments/spec-smoke.yaml`, run via statecraft-serving's
  `bench.harness` with `--results-dir results`): random head loads as an EAGLE drafter with
  guided decoding + prefix caching (0.1% acceptance, as expected; weights +0.10 GiB, so embed and
  lm_head are shared). n-gram (k=3, lookup 2-5) at 4 factions: mean acceptance length 2.32,
  57% draft acceptance, already a strong baseline. Enabling speculation cuts KV cache from 1.95 to
  1.35-1.46 GiB at util 0.8 (bigger CUDA graph pool estimate, 0.44-0.63 GiB): account for it in
  Phase 4 KV predictions and high-faction runs.
- Harness additions in statecraft-serving: `server.volumes` (extra `docker run -v`; spec
  experiments mount `speculative-statecraft_spec-data:/data`, the compose volume's pinned name),
  and a failed server start now removes its container.
- Next: `spec/train.py` (EAGLE-1 losses, seed holdout, per-region metrics). (`--limit 2` first; check
  prompt/completion token-count mismatches and stored size), then `spec/draft_head.py`.

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
