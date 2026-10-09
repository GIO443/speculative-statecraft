## Decode step time (median over pure-decode steps, ms)

| guided | batch | none | ngram-k1 | random-k1 | eagle-k1 | eagle-k3 | ngram-k3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| False | 1 | 16.3 | 17.5 | 21.7 | 23.2 | - | - |
| False | 4 | 16.7 | 26.5 | 26.3 | 27.9 | - | - |
| False | 16 | 20.2 | 43.3 | 44.4 | 46.4 | - | - |
| False | 32 | 24.8 | 66.9 | 68.1 | 68.6 | - | - |
| True | 1 | 15.8 | 17.8 | 22.1 | 23.8 | 27.9 | - |
| True | 4 | 17.6 | 26.9 | 27.0 | 29.1 | 33.6 | - |
| True | 16 | 23.0 | 45.2 | 44.9 | 51.9 | 54.2 | - |
| True | 32 | 27.0 | 69.6 | 68.5 | 73.3 | 83.2 | - |

## Speculative acceptance by workload

| server | workload | mean acceptance length | draft acceptance rate |
|---|---|---:|---:|
| eagle-k3 | game_turn | 3.05 | 0.70 |
| eagle-k3 | ticket_json | 1.34 | 0.19 |
| eagle-k3 | prose | 1.20 | 0.07 |
| ngram-k3 | game_turn | 1.89 | 0.41 |
| ngram-k3 | ticket_json | 2.15 | 0.49 |
| ngram-k3 | prose | 1.22 | 0.07 |

Measured under WSL2 / Docker Desktop on Windows. Pinned host memory is unavailable under WSL (vLLM falls back to device memory for UVA buffers), and Windows reserves VRAM for the desktop; absolute numbers may differ from native Linux.
