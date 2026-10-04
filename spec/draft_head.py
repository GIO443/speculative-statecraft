"""EAGLE-1 draft head, written to match vLLM v0.30's `EagleLlamaForCausalLM` exactly.

At position i the head sees the target's hidden state f_i and the embedding of the next token
x_{i+1}, and produces a feature that should equal f_{i+1}; the target's LM head turns it into a
distribution over x_{i+2}. vLLM computes (model_executor/models/llama_eagle.py):

    x   = fc(cat(embed(x_{i+1}), f_i))            # no bias
    h   = x + attn(x)                             # layer 0's input_layernorm is Identity
    out = h + mlp(post_attention_layernorm(h))    # no final norm
    logits = target.lm_head(out)                  # shared with the target (tied embedding here)

and on later draft steps feeds `out` back in as the feature. Module names mirror vLLM's
checkpoint layout (`fc.weight`, `layers.0.self_attn.q_proj.weight`, ...), so the state dict is
the export. The head owns no embedding or LM head: vLLM shares the target's when the checkpoint
has none, and training uses the target's frozen weights the same way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int

    @classmethod
    def from_target(cls, target: dict[str, Any]) -> HeadConfig:
        """Same widths, heads and RoPE as the target (Qwen2 config.json), so features line up."""
        heads = target["num_attention_heads"]
        return cls(
            hidden_size=target["hidden_size"],
            intermediate_size=target["intermediate_size"],
            num_attention_heads=heads,
            num_key_value_heads=target["num_key_value_heads"],
            head_dim=target.get("head_dim") or target["hidden_size"] // heads,
            rms_norm_eps=target["rms_norm_eps"],
            rope_theta=target.get("rope_theta") or target["rope_parameters"]["rope_theta"],
            vocab_size=target["vocab_size"],
            max_position_embeddings=target["max_position_embeddings"],
        )

    def vllm_config(self) -> dict[str, Any]:
        """config.json for the exported head; vLLM renames LlamaForCausalLM -> EagleLlama..."""
        return {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 1,
            "hidden_act": "silu",
            "attention_bias": False,
            "mlp_bias": False,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta},
            **asdict(self),
        }


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """NeoX-style (rotate-half) RoPE tables, as Llama in vLLM and HF use. positions: [B, T]."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    )
    freqs = positions.float()[..., None] * inv_freq  # [B, T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, heads, T, D]; cos/sin: [B, T, D]."""
    cos, sin = cos[:, None], sin[:, None]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, d = cfg.hidden_size, cfg.head_dim
        self.q_proj = nn.Linear(h, cfg.num_attention_heads * d, bias=False)
        self.k_proj = nn.Linear(h, cfg.num_key_value_heads * d, bias=False)
        self.v_proj = nn.Linear(h, cfg.num_key_value_heads * d, bias=False)
        self.o_proj = nn.Linear(cfg.num_attention_heads * d, h, bias=False)

    def _heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n, self.cfg.head_dim).transpose(1, 2)

    def kv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotated keys and values, [B, kv_heads, T, D]."""
        n = self.cfg.num_key_value_heads
        return _rope(self._heads(self.k_proj(x), n), cos, sin), self._heads(self.v_proj(x), n)

    def attend(
        self,
        xq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Queries from xq against given keys/values; mask [Tq, Tk] (True = attend) or None
        for plain causal attention over a full sequence."""
        b, t, _ = xq.shape
        q = _rope(self._heads(self.q_proj(xq), self.cfg.num_attention_heads), cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=mask is None, enable_gqa=True
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        k, v = self.kv(x, cos, sin)
        return self.attend(x, cos, sin, k, v, None)


class MLP(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Llama decoder layer with the input norm removed (vLLM's EAGLE layer 0)."""

    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = x + self.self_attn(x, cos, sin)
        return h + self.mlp(self.post_attention_layernorm(h))

    def forward_queries(
        self,
        xq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        h = xq + self.self_attn.attend(xq, cos, sin, k, v, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class DraftHead(nn.Module):
    def __init__(self, cfg: HeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layers = nn.ModuleList([DecoderLayer(cfg)])

    def inputs(self, embeds: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat((embeds, features), dim=-1))

    def rope(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta, dtype)

    def forward(
        self, embeds: torch.Tensor, features: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """embeds: embed(x_{i+1}), features: f_i, positions: i. All [B, T, ...]; causal."""
        x = self.inputs(embeds, features)
        cos, sin = self.rope(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x

    # Query-subset path. With one layer, a position's output depends on the other positions
    # only through layer 0's keys/values, which come straight from fc(...). So the context's
    # keys/values can be computed once and outputs evaluated at just the positions that need
    # them (training loss positions, chained draft steps), each with its own attention mask.

    def context(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer-0 keys/values for inputs x = self.inputs(...) at the given positions."""
        cos, sin = self.rope(positions, x.dtype)
        return self.layers[0].self_attn.kv(x, cos, sin)

    def forward_queries(
        self,
        xq: torch.Tensor,
        positions: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Outputs for query inputs xq at `positions`, attending to k/v under mask [Tq, Tk]."""
        cos, sin = self.rope(positions, xq.dtype)
        return self.layers[0].forward_queries(xq, cos, sin, k, v, mask)
