"""Field-Aware Feed-Forward Network for FAT (Field-Aware Transformer).

Paper §3.3: Replaces the standard shared FFN with field-conditioned FFN:

    FAT-FFN(z_i) = SiLU(z_i · W_1^{(f_i)}) · W_2^{(f_i)}

Each field f has dedicated projection matrices W_1^{(f)} and W_2^{(f)},
synthesised by the Basis-Composed Hypernetwork.

Reference:
    https://arxiv.org/abs/2511.12081
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldAwareFFN(nn.Module):
    """Field-conditioned FFN with per-field W₁, W₂ projections.

    Args:
        d: Model dimension.
        d_ff: FFN intermediate dimension.
    """

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.d = d
        self.d_ff = d_ff

        # LayerNorm applied before FFN (Pre-LN style)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        W_1: torch.Tensor,
        W_2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with pre-computed field-specific projections.

        Args:
            x: [batch, num_fields, d]  input token embeddings.
            W_1: [num_fields, d, d_ff]  field-specific W₁.
            W_2: [num_fields, d_ff, d]  field-specific W₂.

        Returns:
            output: [batch, num_fields, d]
        """
        # Pre-LN
        x = self.norm(x)

        # FAT-FFN: SiLU(x · W₁) · W₂
        #   x: [B, F, d],  W_1: [F, d, d_ff]  →  hidden: [B, F, d_ff]
        hidden = torch.einsum("bfd,fde->bfe", x, W_1)
        hidden = F.silu(hidden)
        #   hidden: [B, F, d_ff],  W_2: [F, d_ff, d]  →  output: [B, F, d]
        output = torch.einsum("bfe,fed->bfd", hidden, W_2)

        return output
