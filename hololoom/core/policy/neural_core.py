"""
Neural Network Components for Policy Engine
=============================================
Motif-gated attention, LoRA adapters, and NeuralCore decision network.

Extracted from unified.py (March 2026 Refactor).
"""

from __future__ import annotations

import logging
import math
import time as _time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hololoom.dark_trace.sae.activation_buffer import ActivationSample, get_activation_buffer

logger = logging.getLogger(__name__)


def maybe_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomMHA(nn.Module):
    """Custom Multi-Head Attention with gate control."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.d_head

        q = self.Wq(x).view(B, T, H, Dh)
        k = self.Wk(x).view(B, T, H, Dh)
        v = self.Wv(x).view(B, T, H, Dh)

        attn = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(Dh)
        A = torch.softmax(attn, dim=-1)

        g = gates.view(B, H, 1, 1)
        A = A * g

        z = torch.einsum('bhts,bshd->bthd', A, v).contiguous().view(B, T, D)

        return self.Wo(z), A


class CrossAttention(nn.Module):
    """Cross-attention between query and memory."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        M = mem.size(1)
        H = self.n_heads
        Dh = self.d_head

        q = self.Wq(x).view(B, T, H, Dh)
        k = self.Wk(mem).view(B, M, H, Dh)
        v = self.Wv(mem).view(B, M, H, Dh)

        attn = torch.einsum('bthd,bmhd->bhtm', q, k) / math.sqrt(Dh)
        A = torch.softmax(attn, dim=-1)

        z = torch.einsum('bhtm,bmhd->bthd', A, v).contiguous().view(B, T, D)

        return self.Wo(z)


class MotifGatedMHA(nn.Module):
    """Multi-head attention with motif-based gating."""

    def __init__(self, d_model: int, n_heads: int = 4, n_motifs: int = 8):
        super().__init__()
        self.mha = CustomMHA(d_model, n_heads)
        self.gate_proj = nn.Linear(n_motifs, n_heads)

    def forward(self, x: torch.Tensor, motif_ctrl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate_proj(motif_ctrl))
        out, attn = self.mha(x, gates)
        return out, attn


class LoRALikeFFN(nn.Module):
    """Feed-forward network with LoRA-style adapters."""

    def __init__(self, d_model: int, d_ff: int = 1024, r: int = 8, n_adapters: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, r, bias=False),
                nn.Linear(r, d_model, bias=False)
            )
            for _ in range(n_adapters)
        ])

    def forward(self, x: torch.Tensor, adapter_idx: int = 0) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        h = h + self.adapters[adapter_idx](x)
        return h


class TinyTransformerBlock(nn.Module):
    """Transformer block with cross-attention, motif-gated self-attention, and LoRA FFN."""

    def __init__(self, d_model: int = 384, n_heads: int = 4, n_motifs: int = 8, n_adapters: int = 4):
        super().__init__()
        self.cross = CrossAttention(d_model, n_heads)
        self.mha = MotifGatedMHA(d_model, n_heads, n_motifs)
        self.ffn = LoRALikeFFN(d_model, d_ff=4 * d_model, r=16, n_adapters=n_adapters)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        motif_ctrl: torch.Tensor,
        adapter_idx: int
    ) -> torch.Tensor:
        x = x + self.cross(self.ln1(x), mem)
        mha_out, _ = self.mha(self.ln2(x), motif_ctrl)
        x = x + mha_out
        x = x + self.ffn(self.ln3(x), adapter_idx)
        return x


class NeuralCore(nn.Module):
    """
    Neural decision engine using transformer architecture.

    Uses learnable latent query tokens, stacked transformer blocks with
    cross-attention to context, and a tool selection head.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 2,
        n_heads: int = 4,
        n_motifs: int = 8,
        n_adapters: int = 4,
        n_tools: int = 4,
        semantic_dim: int = 8
    ):
        super().__init__()

        self.latent = nn.Parameter(torch.randn(1, 16, d_model) / math.sqrt(d_model))

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, n_motifs, n_adapters)
            for _ in range(n_layers)
        ])

        self.readout = nn.Linear(d_model, d_model)
        self.tool_head = nn.Linear(d_model, n_tools)

        self.semantic_proj = nn.Linear(semantic_dim, d_model)
        self.semantic_gate = nn.Linear(d_model * 2, d_model)

        self.tools = ["answer", "search", "notion_write", "calc"]

    async def decide(
        self,
        mem: torch.Tensor,
        ctrl: torch.Tensor,
        adapter_idx: int,
        semantic_features: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make a decision given context and control signals."""
        B = mem.size(0)

        x = self.latent.expand(B, -1, -1)

        for blk in self.blocks:
            x = blk(x, mem, ctrl, adapter_idx)

        pooled = x.mean(dim=1)

        if semantic_features is not None:
            semantic_proj = self.semantic_proj(semantic_features)
            combined = torch.cat([pooled, semantic_proj], dim=-1)
            gate = torch.sigmoid(self.semantic_gate(combined))
            pooled = pooled * (1 - gate) + semantic_proj * gate

        logits = self.tool_head(self.readout(pooled))

        _buf = get_activation_buffer()
        if _buf is not None:
            _buf.record_sync(ActivationSample(
                timestamp=_time.time(),
                source="neural_core",
                activation=pooled.detach().cpu().numpy().squeeze(),
                tool_selected=self.tools[logits.argmax().item()],
            ))

        return logits, pooled
