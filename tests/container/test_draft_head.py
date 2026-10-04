"""Draft head vs HF Llama reference, and export format. Trainer container only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safetensors.torch import load_file  # noqa: E402
from transformers.models.llama import modeling_llama  # noqa: E402

from spec.draft_head import DraftHead, HeadConfig, rope_cos_sin  # noqa: E402
from spec.export import export  # noqa: E402

TARGET = {  # Qwen2.5-1.5B-shaped, scaled down
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "vocab_size": 100,
    "max_position_embeddings": 512,
}


def _cfg() -> HeadConfig:
    return HeadConfig.from_target(TARGET)


def _llama_config(cfg: HeadConfig) -> transformers.LlamaConfig:
    c = transformers.LlamaConfig(**cfg.vllm_config())
    c._attn_implementation = "sdpa"
    return c


def test_head_config_from_target() -> None:
    cfg = _cfg()
    assert cfg.head_dim == 16
    vc = cfg.vllm_config()
    assert vc["architectures"] == ["LlamaForCausalLM"] and vc["num_hidden_layers"] == 1
    assert vc["rope_parameters"]["rope_theta"] == 1_000_000.0


def test_rope_matches_hf() -> None:
    cfg = _cfg()
    rotary = modeling_llama.LlamaRotaryEmbedding(_llama_config(cfg))
    pos = torch.tensor([[0, 1, 5, 300]])
    x = torch.zeros(1, 4, cfg.hidden_size)
    hf_cos, hf_sin = rotary(x, pos)
    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, torch.float32)
    torch.testing.assert_close(cos, hf_cos)
    torch.testing.assert_close(sin, hf_sin)


def test_layer_matches_hf_llama_without_input_norm() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    head = DraftHead(cfg).eval()
    for p in head.parameters():  # non-trivial norm weights too
        torch.nn.init.normal_(p, std=0.1)
    lc = _llama_config(cfg)
    ref = modeling_llama.LlamaDecoderLayer(lc, layer_idx=0).eval()
    ref.input_layernorm = torch.nn.Identity()
    missing, unexpected = ref.load_state_dict(head.layers[0].state_dict(), strict=False)
    assert not unexpected and set(missing) <= {"input_layernorm.weight"}

    x = torch.randn(2, 7, cfg.hidden_size)
    pos = torch.arange(3, 10).expand(2, 7)
    cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, x.dtype)
    with torch.no_grad():
        ours = head.layers[0](x, cos, sin)
        theirs = ref(x, attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
    theirs = theirs[0] if isinstance(theirs, tuple) else theirs
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-5)


def test_head_is_causal() -> None:
    torch.manual_seed(0)
    cfg = _cfg()
    head = DraftHead(cfg).eval()
    e, f = torch.randn(1, 6, cfg.hidden_size), torch.randn(1, 6, cfg.hidden_size)
    pos = torch.arange(6)[None]
    with torch.no_grad():
        full = head(e, f, pos)
        e2, f2 = e.clone(), f.clone()
        e2[:, 4:], f2[:, 4:] = 9.0, -9.0  # change the future only
        changed = head(e2, f2, pos)
    torch.testing.assert_close(full[:, :4], changed[:, :4])


def test_export_layout(tmp_path: Path) -> None:
    head = DraftHead(_cfg())
    export(head, tmp_path)
    tensors = load_file(str(tmp_path / "model.safetensors"))
    layer = "layers.0."
    assert set(tensors) == {
        "fc.weight",
        layer + "self_attn.q_proj.weight",
        layer + "self_attn.k_proj.weight",
        layer + "self_attn.v_proj.weight",
        layer + "self_attn.o_proj.weight",
        layer + "post_attention_layernorm.weight",
        layer + "mlp.gate_proj.weight",
        layer + "mlp.up_proj.weight",
        layer + "mlp.down_proj.weight",
    }
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())
    assert tensors["fc.weight"].shape == (64, 128)
    config = json.loads((tmp_path / "config.json").read_text("utf-8"))
    assert transformers.LlamaConfig(**config).num_hidden_layers == 1
