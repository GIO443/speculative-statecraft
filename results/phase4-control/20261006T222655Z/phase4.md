| variant | factions | s/turn | speedup | MAL | per-position | KV GiB | excluded |
|---|---:|---:|---:|---:|---|---:|---:|
| baseline | 4 | 4.58 ± 0.33 | 1.00 | - | - | 1.93 | 0 |
| baseline | 8 | 5.40 ± 0.10 | 1.00 | - | - | 1.93 | 0 |
| baseline | 16 | 6.53 ± 0.21 | 1.00 | - | - | 1.93 | 0 |
| baseline | 32 | 8.76 ± 0.19 | 1.00 | - | - | 1.93 | 0 |
| baseline | 64 | 14.12 ± 0.22 | 1.00 | - | - | 1.93 | 0 |
| eagle1-k2 | 4 | 3.81 ± 0.20 | 1.20 | 2.13 | 0.66 / 0.47 | 1.41 | 0 |
| eagle1-k2 | 8 | 5.02 ± 0.99 | 1.07 | 2.18 | 0.67 / 0.51 | 1.41 | 0 |
| eagle1-k2 | 16 | 6.95 ± 0.07 | 0.94 | 2.27 | 0.71 / 0.55 | 1.41 | 0 |
| eagle1-k2 | 32 | 9.36 ± 1.03 | 0.94 | 2.36 | 0.75 / 0.62 | 1.41 | 0 |
| eagle1-k2 | 64 | 15.51 ± 0.60 | 0.91 | 2.40 | 0.76 / 0.64 | 1.41 | 0 |

Measured under WSL2 / Docker Desktop on Windows. Pinned host memory is unavailable under WSL (vLLM falls back to device memory for UVA buffers), and Windows reserves VRAM for the desktop; absolute numbers may differ from native Linux.
