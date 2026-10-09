## Decode step time (median over pure-decode steps, ms)

| guided | batch | none | ngram-k1 |
|---|---:|---:|---:|
| False | 1 | 17.1 | 18.1 |
| False | 4 | 17.2 | 23.3 |
| False | 16 | 19.6 | 33.9 |
| False | 32 | 22.4 | 48.1 |

## Speculative acceptance by workload

| server | workload | mean acceptance length | draft acceptance rate |
|---|---|---:|---:|

Measured under WSL2 / Docker Desktop on Windows. Pinned host memory is unavailable under WSL (vLLM falls back to device memory for UVA buffers), and Windows reserves VRAM for the desktop; absolute numbers may differ from native Linux.
