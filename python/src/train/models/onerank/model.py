"""OneRank: Unified Transformer-Native Ranking Architecture — main model.

Paper: arXiv:2606.16838  (KDD 2026)

Architecture:
  1. FeatureEmbeddings → per-field token embeddings  [B, F, d]
  2. Task token injection → [B, F+K, d] with structured mask
  3. L × OneRankBlock (Pre-LN MHSA + Pre-LN FFN)
  4. Task representation extraction + feature pooling
  5. Task-specific SD projection
  6. Cross-task attention (configurable mask)
  7. Dynamic matching scoring: s_k = z_k^T · r_k
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
from .encoding import OneRankBlock, build_attention_mask
from .prediction import CrossTaskAttention, matching_score


class OneRankModel(nn.Module):
    """OneRank: Unified Transformer-Native Ranking Architecture.

    Args:
        features: List of (name, vocab_size, embed_dim) feature specs.
        d: Model (hidden) dimension.
        d_ff: FFN intermediate dimension.
        num_layers: Number of Transformer blocks (L).
        n_heads: Number of attention heads.
        num_tasks: Number of tasks K.
        cross_task_mask: Mask type for cross-task attention.
        dropout: Dropout rate.
        pooling_map: Per-feature pooling strategy override.
        total_dim: Total embedding dimension override.
        output_contract: NormalizedOutputContract (mutually exclusive with task_config).
    """

    def __init__(
        self,
        features: list[FeatureTuple],
        d: int = 128,
        d_ff: int = 512,
        num_layers: int = 2,
        n_heads: int = 8,
        num_tasks: int = 3,
        cross_task_mask: str = "cascade",
        dropout: float = 0.0,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        self.d = d
        self.num_tasks = num_tasks

        # Feature embeddings
        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        num_fields = len(features)

        # Task token parameters: [K, d]
        self.task_tokens = nn.Parameter(torch.randn(num_tasks, d) * 0.02)

        # Learned positional encoding: [1, N_max, d]
        self.pos_encoding = nn.Parameter(torch.randn(1, num_fields + num_tasks, d) * 0.02)

        # Input projection (if feature dims differ from model dim)
        embed_dim = features[0][2] if features else d
        if any(f[2] != embed_dim for f in features) or embed_dim != d:
            self.input_proj = nn.Linear(embed_dim, d)
        else:
            self.input_proj = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            OneRankBlock(d, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # Task-specific situational descriptor projection
        self.sd_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )

        # Cross-task attention
        self.cross_task = CrossTaskAttention(d, num_tasks, cross_task_mask)

        if output_contract is not None:
            self.output_contract = output_contract
            self.task_names = [tower.name for tower in output_contract.towers]
        else:
            self.task_names = [f"task_{i}" for i in range(num_tasks)]

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        """Forward pass returning ModelOutput."""
        if hasattr(self, "output_contract"):
            return self.forward_execution(x_inputs).outputs
        scores, _ = self._shared(x_inputs)
        outputs = ModelOutput()
        for k, name in enumerate(self.task_names):
            outputs.insert(name, scores[:, k], OutputKind.BinaryLogit)
        return outputs

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        """Forward pass returning full execution graph."""
        scores, _ = self._shared(x_inputs)
        if hasattr(self, "output_contract"):
            outputs = ModelOutput()
            for k, name in enumerate(self.task_names):
                outputs.insert(name, scores[:, k], OutputKind.BinaryLogit)
            return ModelExecution(nodes=outputs, outputs=outputs)
        outputs = ModelOutput()
        for k, name in enumerate(self.task_names):
            outputs.insert(name, scores[:, k], OutputKind.BinaryLogit)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute matching scores and task representations.

        Returns:
            scores: [B, K] per-task matching scores.
            task_repr: [B, K, d] task token representations.
        """
        stacked = self.embeddings.forward_stacked(x_inputs)
        x = torch.cat(stacked, dim=1)
        if self.input_proj is not None:
            x = self.input_proj(x)

        B, F, _ = x.shape
        K = self.num_tasks

        # Inject task tokens: [B, K, d]
        task_tokens = self.task_tokens[None, :, :].expand(B, -1, -1)
        h = torch.cat([x, task_tokens], dim=1)

        # Learned positional encoding
        h = h + self.pos_encoding[:, : F + K, :]

        # Build structured attention mask bias
        mask = build_attention_mask(F, K, h.device)
        mask_bias = (1.0 - mask) * float("-inf")

        # Transformer blocks
        for block in self.blocks:
            h = block(h, mask_bias)

        # Extract task representations: [B, K, d]
        task_repr = h[:, F:, :]

        # Feature pool → situational descriptor: [B, d]
        feat_pool = h[:, :F, :].mean(dim=1)
        sd = self.sd_proj(feat_pool)

        # Expand SD per task: [B, d] → [B, K, d]
        sd = sd[:, None, :].expand(-1, K, -1)

        # Cross-task attention on task_repr + SD
        cross_input = task_repr + sd
        global_repr = self.cross_task(cross_input)

        # Matching scoring: s_k = z_k^T · r_k
        scores = matching_score(global_repr, task_repr)

        return scores, task_repr
