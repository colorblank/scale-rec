"""OneRank prediction head: cross-task attention + matching scoring.

Paper §2.4: Controlled cross-task knowledge transfer with configurable
masking and dynamic matching-based scoring.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossTaskAttention(nn.Module):
    """Cross-task self-attention with configurable attention mask.

    Paper §2.4 Eq. (9)–(10): Task representations attend to each other
    according to a configurable mask, then residual + FFN.

    Args:
        d: Model dimension.
        num_tasks: Number of tasks K.
        mask_type: One of "parallel", "null", "cascade".
    """

    def __init__(self, d: int, num_tasks: int, mask_type: str = "cascade"):
        super().__init__()
        self.d = d
        self.num_tasks = num_tasks
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )

        if mask_type == "parallel":
            mask = torch.eye(num_tasks)
        elif mask_type == "null":
            mask = torch.ones(num_tasks, num_tasks)
        elif mask_type == "cascade":
            mask = torch.tril(torch.ones(num_tasks, num_tasks))
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
        self.register_buffer("cross_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: cross-task attention + residual + FFN.

        Args:
            x: [B, K, d] task representations.

        Returns: [B, K, d]
        """
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d**0.5)
        bias = (1.0 - self.cross_mask) * float("-inf")
        scores = scores + bias[None, :, :]
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v)
        x = residual + x

        residual2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual2 + x
        return x


def matching_score(global_repr: torch.Tensor, task_repr: torch.Tensor) -> torch.Tensor:
    """Dynamic matching-based scoring: s_k = z_k^T · r_k

    Paper §2.4 Eq. (12): Inner product between task-aware global context
    and context-conditioned candidate embeddings.

    Args:
        global_repr: [B, K, d] z_k (post cross-task attention).
        task_repr: [B, K, d] r_k (from task token extraction).

    Returns: [B, K] per-task scores.
    """
    return (global_repr * task_repr).sum(dim=-1)
