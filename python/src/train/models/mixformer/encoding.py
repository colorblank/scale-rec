"""MixFormer core modules: HeadMixing, SwiGLU FFN, QueryMixer, OutputFusion.

Paper §3.3: Each MixFormer block consists of QueryMixer + CrossAttention + OutputFusion.
Here we implement QueryMixer and OutputFusion with per-head SwiGLU FFN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def head_mixing(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free cross-head information exchange.

    Paper §3.3.1 Eq. (3): Reshapes [B, N, D] → [B, N, N, D/N],
    transposes dims 1↔2, and flattens to [B, N, D].

    Each output head j receives chunk j from every input head,
    enabling cross-head communication at zero parameter cost.
    """
    B, N, D = x.shape
    return x.reshape(B, N, N, D // N).transpose(1, 2).reshape(B, N, D)


class SwiGLUFFN(nn.Module):
    """SwiGLU-activated feed-forward network.

    Paper §3.3: SwiGLU(x) = (SiLU(x · W_gate) ⊗ (x · W_up)) · W_down
    where ⊗ denotes element-wise multiplication.
    """

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class QueryMixer(nn.Module):
    """Query Mixer: replaces self-attention with HeadMixing + per-head SwiGLU FFN.

    Paper §3.3.1 Eq. (3)–(4):
        P = HeadMixing(Norm(X)) + X
        Q_i = SwiGLUFFN_i(Norm(P_i)) + P_i
    """

    def __init__(self, d: int, d_ff: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.head_ffns = nn.ModuleList([SwiGLUFFN(d, d_ff) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, N, _ = x.shape

        residual = x
        x = self.norm1(x)
        x = head_mixing(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        outputs = [self.head_ffns[i](x[:, i, :]) for i in range(N)]
        x = torch.stack(outputs, dim=1)
        x = x + residual
        return x


class OutputFusion(nn.Module):
    """Output Fusion: per-head SwiGLU FFN + residual.

    Paper §3.3.3 Eq. (9):
        o_i = SwiGLUFFN_i(Norm(z_i)) + z_i

    Each head receives its own FFN after cross-attention aggregation.
    """

    def __init__(self, d: int, d_ff: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.head_ffns = nn.ModuleList([SwiGLUFFN(d, d_ff) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, N, _ = x.shape

        residual = x
        x = self.norm(x)
        outputs = [self.head_ffns[i](x[:, i, :]) for i in range(N)]
        x = torch.stack(outputs, dim=1)
        x = x + residual
        return x


class MixFormerBlock(nn.Module):
    """One MixFormer block: QueryMixer + OutputFusion with Pre-LN.

    Paper §3.3:
        Z^{(l)} = OutputFusion(CrossAttn(QueryMixer(Z^{(l-1)})))
    """

    def __init__(self, d: int, d_ff: int, num_heads: int):
        super().__init__()
        self.query_mixer = QueryMixer(d, d_ff, num_heads)
        self.output_fusion = OutputFusion(d, d_ff, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.query_mixer(x)
        x = self.output_fusion(x)
        return x
