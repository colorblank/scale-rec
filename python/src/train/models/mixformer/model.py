"""MixFormer: Co-Scaling Up Dense and Sequence — main model.

Paper: arXiv:2602.14110  (KDD 2026)

Architecture:
  1. FeatureEmbeddings → per-field embeddings
  2. Concatenate + project → [B, N, D] (N query heads)
  3. L × MixFormerBlock (QueryMixer → OutputFusion)
  4. Mean pool heads → [B, D] → task towers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...core.config import PoolingMode

from ...core.model_output import ModelExecution, ModelOutput, OutputKind
from ...core.output_contract import NormalizedOutputContract
from ...layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..output_head import OutputHead
from .encoding import MixFormerBlock


class MixFormerModel(nn.Module):
    """MixFormer: Unified Transformer for dense + sequence co-scaling.

    Args:
        features: List of (name, vocab_size, embed_dim) feature specs.
        d: Head dimension D (paper: 386 small, 768 medium).
        d_ff: SwiGLU FFN hidden dimension per head.
        num_heads: Number of query heads N (paper: 16).
        num_layers: Number of MixFormer blocks L (paper: 4).
        dropout: Dropout rate.
        pooling_map: Per-feature pooling strategy override.
        total_dim: Total embedding dimension override.
        output_contract: NormalizedOutputContract.
    """

    def __init__(
        self,
        features: list[FeatureTuple],
        d: int = 386,
        d_ff: int = 1024,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.0,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.num_heads = num_heads

        # Feature embeddings
        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)

        # Compute total non-sequential embedding dim
        total_dim_sum = total_dim or sum(f[2] for f in features)

        # Project concatenated features to [B, N*D]
        self.input_proj = nn.Linear(total_dim_sum, num_heads * d)

        # MixFormer blocks
        self.blocks = nn.ModuleList([MixFormerBlock(d, d_ff, num_heads) for _ in range(num_layers)])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )

        if output_contract is not None:
            self.output_contract = output_contract

            self.output_head = OutputHead(
                output_contract,
                {"shared": d},
            )
            self.task_names = [tower.name for tower in output_contract.towers]
        else:
            self.task_names = [f"task_{i}" for i in range(3)]

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_contract"):
            return self.forward_execution(x_inputs).outputs
        outputs = ModelOutput()
        scores = self._shared(x_inputs)
        name = self.task_names[0]
        outputs.insert(name, scores, OutputKind.BinaryLogit)
        return outputs

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        shared = self._shared(x_inputs)
        if hasattr(self, "output_contract"):
            return self.output_head.forward({"shared": shared})
        outputs = ModelOutput()
        name = self.task_names[0]
        outputs.insert(name, shared, OutputKind.BinaryLogit)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        """Compute shared representation: [B, D]"""
        # Stack per-field embeddings and project
        stacked = self.embeddings.forward_stacked(x_inputs)
        x = torch.cat(stacked, dim=1)  # [B, F, total_dim_sum/F_avg]
        x = x.mean(dim=1)  # [B, total_dim_sum]

        # Project to multi-head query space
        x = self.input_proj(x)  # [B, N*D]
        B, _ = x.shape
        x = x.view(B, self.num_heads, self.d)  # [B, N, D]

        # MixFormer blocks
        for block in self.blocks:
            x = block(x)

        # Pool heads → [B, D]
        x = x.mean(dim=1)  # [B, D]
        x = self.output_proj(x)
        return x
