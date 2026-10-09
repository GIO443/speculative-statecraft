## Decode step time (median over pure-decode steps, ms)

| guided | batch | none-flash | ngram-k1-flash | none-flashinfer | ngram-k1-flashinfer | none-triton | ngram-k1-triton |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 1 | 16.9 | 17.8 | 16.2 | 17.6 | 16.5 | 16.8 |
| False | 16 | 20.9 | 43.3 | 18.6 | 26.3 | 20.1 | 24.3 |
| False | 32 | 24.5 | 67.2 | 20.3 | 29.8 | 23.1 | 28.9 |

## Speculative acceptance by workload

| server | workload | mean acceptance length | draft acceptance rate |
|---|---|---:|---:|

Measured under WSL2 / Docker Desktop on Windows. Pinned host memory is unavailable under WSL (vLLM falls back to device memory for UVA buffers), and Windows reserves VRAM for the desktop; absolute numbers may differ from native Linux.
