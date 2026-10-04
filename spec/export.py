"""Export a draft head as a checkpoint vLLM v0.30 loads with `{"method": "eagle", "model": DIR}`.

    python -m spec.export --target Qwen/Qwen2.5-1.5B-Instruct --out /data/heads/<name> \
        [--checkpoint train_state.pt | --random]

Writes config.json (Llama architecture; vLLM wraps it as EagleLlamaForCausalLM) and
model.safetensors with exactly the head's own weights. No embed_tokens / lm_head are written,
so vLLM shares the target's. `--random` exports an untrained head, used to check that the
format loads before spending GPU time on training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from spec.draft_head import DraftHead, HeadConfig

DTYPE = torch.bfloat16


def export(head: DraftHead, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().to(DTYPE).contiguous().cpu() for k, v in head.state_dict().items()}
    save_file(tensors, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(head.cfg.vllm_config(), indent=2), "utf-8")


def target_config(model: str) -> dict:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model).to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="training checkpoint with a `head` state")
    src.add_argument("--random", action="store_true", help="untrained head (format check)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    head = DraftHead(HeadConfig.from_target(target_config(args.target)))
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        head.load_state_dict(state["head"])
    export(head, args.out)
    n = sum(p.numel() for p in head.parameters())
    print(f"exported {n / 1e6:.1f}M-parameter head to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
