"""OneRank Transformer block with structured attention masking.

Paper §2.1–§2.2: Structured tokenization with task-specific tokens
and mutual invisibility masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_attention_mask(num_fields: int, num_tasks: int, device: torch.device) -> torch.Tensor:
    """Build structured attention mask for OneRank.

    Shape [N, N] where N = F + K:
      - Features ↔ Features: 1 (bidirectional)
      - Features → Task tokens: 0
      - Task tokens → Features: 1
      - Task token k → Task token k: 1 (self)
      - Task token k → Task token j (j≠k): 0 (mutual invisibility)

    Returns a float mask: 1.0 = allowed, 0.0 = masked.
    """
    N = num_fields + num_tasks
    mask = torch.zeros(N, N, device=device)
    mask[:num_fields, :num_fields] = 1.0
    mask[num_fields:, :num_fields] = 1.0
    for k in range(num_tasks):
        mask[num_fields + k, num_fields + k] = 1.0
    return mask


class OneRankBlock(nn.Module):
    """One Transformer block for OneRank: Pre-LN MHSA + Pre-LN FFN.

    Uses separate Q/K/V Linear layers (not nn.MultiheadAttention) for
    clean weight naming parity with the Rust Candle implementation:
      attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight

    Args:
        d: Model dimension.
        n_heads: Number of attention heads.
        d_ff: FFN intermediate dimension.
        dropout: Dropout rate.
    """

    def __init__(self, d: int, n_heads: int = 8, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        ffn_dim = d_ff if d_ff is not None else d * 4

        self.norm1 = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        """Forward with attention bias (float mask).

        Multi-head dot-product attention with separate Q/K/V projections,
        matching the Candle Rust implementation.

        Args:
            x: [B, N, d]
            mask_bias: [N, N] where 0.0 = allowed, -inf = masked.

        Returns: [B, N, d]
        """
        B, N, _ = x.shape

        residual = x
        x = self.norm1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head: [B, n_heads, N, d_head]
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_head**-0.5)
        scores = scores + mask_bias[None, None, :, :]

        attn = torch.softmax(scores, dim=-1)
        attn = torch.dropout(attn, self.dropout.p, train=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d)
        out = self.o_proj(out)
        x = residual + out

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x
