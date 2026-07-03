"""Field-Decomposed Attention for FAT (Field-Aware Transformer).

Paper §3.2: Replaces standard self-attention with two field-aware mechanisms:

  (a) Field-aware content alignment — each field f has dedicated
      Q/K/V projection matrices W_Q^{(f)}, W_K^{(f)}, W_V^{(f)}.

  (b) Field-pair interaction modulation — a learnable scalar w_{f_i,f_j}
      gates the attention score between field i and field j.

The score from token i (field f_i) to token j (field f_j):

    s(i,j) = (q_i^T k_j) · w_{f_i, f_j}

The output uses standard softmax attention with the modulated scores.
Multi-head splitting is applied on the projected Q/K/V tensors.

Reference:
    https://arxiv.org/abs/2511.12081
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FieldDecomposedAttention(nn.Module):
    """Single-head or multi-head field-aware attention layer.

    Args:
        d: Model dimension.
        n_heads: Number of attention heads (default 8).
        num_fields: Number of fields F, for field-pair interaction modulation.
        dropout: Attention dropout (default 0.0).
    """

    def __init__(
        self,
        d: int,
        n_heads: int = 8,
        num_fields: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.dropout = dropout

        # LayerNorm applied before attention (Pre-LN style)
        self.norm = nn.LayerNorm(d)

        # Learnable field-aware bias: b_f ∈ ℝ^d  (paper §3.1)
        if num_fields > 0:
            self.field_bias = nn.Parameter(torch.zeros(num_fields, d))

    def forward(
        self,
        x: torch.Tensor,
        W_q: torch.Tensor,
        W_k: torch.Tensor,
        W_v: torch.Tensor,
        field_pair_w: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with pre-computed field-specific projections.

        Args:
            x: [batch, num_fields, d]  input token embeddings.
            W_q: [num_fields, d, d]  field-specific Q projections.
            W_k: [num_fields, d, d]  field-specific K projections.
            W_v: [num_fields, d, d]  field-specific V projections.
            field_pair_w: [num_fields, num_fields]  interaction modulation.

        Returns:
            output: [batch, num_fields, d]
        """
        B, F, D = x.shape

        # Pre-LN
        x = self.norm(x)

        # Add field-aware bias:  h_i = e_i + b_{f_i}
        if hasattr(self, "field_bias"):
            x = x + self.field_bias.unsqueeze(0)  # [B, F, d]

        # Per-field Q/K/V projections:  q_i = h_i · W_Q^{(f_i)}
        #   Q = einsum("bfd,fde->bfe", x, W_q)  → [B, F, d]
        Q = torch.einsum("bfd,fde->bfe", x, W_q)
        K = torch.einsum("bfd,fde->bfe", x, W_k)
        V = torch.einsum("bfd,fde->bfe", x, W_v)

        # Reshape to multi-head: [B, F, n_heads, d_head] → [B, n_heads, F, d_head]
        Q = Q.view(B, F, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, F, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, F, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled attention scores
        scores = torch.einsum("bnid,bnjd->bnij", Q, K) * (self.d_head ** -0.5)

        # Field-pair interaction modulation:  s(i,j) *= w_{f_i, f_j}
        scores = scores * field_pair_w[None, None, :, :]  # [1, 1, F, F]

        # Softmax + dropout
        attn = torch.softmax(scores, dim=-1)
        attn = torch.dropout(attn, self.dropout, train=self.training)

        # Weighted sum over values
        out = torch.einsum("bnij,bnjd->bnid", attn, V)  # [B, n_heads, F, d_head]
        out = out.transpose(1, 2).contiguous().view(B, F, D)

        return out
