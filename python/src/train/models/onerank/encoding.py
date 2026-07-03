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
    mask[num_tasks:, :num_fields] = 1.0
    for k in range(num_tasks):
        mask[num_fields + k, num_fields + k] = 1.0
    return mask


class OneRankBlock(nn.Module):
    """One Transformer block for OneRank: Pre-LN MHSA + Pre-LN FFN.

    Args:
        d: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"

        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor) -> torch.Tensor:
        """Forward with attention bias (float mask).

        Args:
            x: [B, N, d]
            mask_bias: [N, N] where 0.0 = allowed, -inf = masked.

        Returns: [B, N, d]
        """
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, attn_mask=mask_bias, need_weights=False)[0]
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x
