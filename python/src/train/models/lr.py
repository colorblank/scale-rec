from __future__ import annotations

"""逻辑回归基线：Embedding + Linear。"""
import torch.nn as nn

from ..layers.embedding import FeatureEmbeddings
from ..layers.mlp import Mlp
from ..layers.towers import Activation


class LogisticRegression(nn.Module):
    """LR baseline: Embedding + Linear, no feature interaction."""

    def __init__(self, features):
        """Build LR: FeatureEmbeddings + Mlp([], 1, NONE)."""
        super().__init__()
        self.embeddings = FeatureEmbeddings(features)
        self.mlp = Mlp(self.embeddings.total_dim, [], 1, Activation.NONE)

    def forward(self, x_inputs):
        """Forward: embed -> concat -> linear -> {"pred": logits}."""
        return {"pred": self.mlp(self.embeddings(x_inputs))}
