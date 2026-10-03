# speculative-statecraft

Phases 2-4 of the statecraft project: train an EAGLE-style draft head on the
[statecraft-serving](https://github.com/GIO443/statecraft-serving) simulation's own outputs and
compare it with no speculation, n-gram speculation and a generic drafter, across faction counts.

## Setup

Clone next to `statecraft-serving` (it is an editable path dependency), then:

```powershell
uv sync
uv run pytest
```
