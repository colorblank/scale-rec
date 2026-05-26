from __future__ import annotations

"""逻辑回归基线：Embedding + Linear。"""
import torch
import torch.nn as nn

from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation


class LogisticRegression(nn.Module):
    """LR baseline: Embedding + Linear, no feature interaction."""

    def __init__(
        self,
        features: list[FeatureTuple],
        pooling_map: dict[str, str] | None = None,
        total_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        self.mlp = Mlp(self.embeddings.total_dim, [], 1, Activation.NONE)

    def forward(self, x_inputs: FeatureTensorMap) -> dict[str, torch.Tensor]:
        """Forward: embed -> concat -> linear -> {"pred": logits}."""
        return {"pred": self.mlp(self.embeddings(x_inputs))}
