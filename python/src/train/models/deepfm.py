from __future__ import annotations

"""DeepFM：FM 一阶 + FM 二阶 + Deep MLP。"""
import torch
import torch.nn as nn

from ..layers.embedding import FeatureEmbeddings
from ..layers.fm import fm_interaction
from ..layers.mlp import Mlp
from ..layers.towers import Activation


class DeepFM(nn.Module):
    """DeepFM: FM first-order + FM second-order + Deep MLP."""

    def __init__(self, features, fm_k, deep_hidden_dims, pooling_map=None):
        super().__init__()
        self.fm_first = FeatureEmbeddings([(n, v, 1) for n, v, _ in features], pooling_map)
        self.fm_second = FeatureEmbeddings([(n, v, fm_k) for n, v, _ in features], pooling_map)
        self.fm_k = fm_k
        self.deep = FeatureEmbeddings(features, pooling_map)
        self.deep_total_dim = self.deep.total_dim
        self.deep_mlp = Mlp(self.deep_total_dim, deep_hidden_dims, 1, Activation.RELU)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_inputs):
        """Forward: FM first + FM second + Deep MLP + global_bias -> {"pred": logits}."""
        first = self.fm_first(x_inputs).sum(dim=1, keepdim=True)
        stacked = torch.cat(self.fm_second.forward_stacked(x_inputs), dim=1)
        second = fm_interaction(stacked)
        deep_out = self.deep_mlp(self.deep(x_inputs))
        return {"pred": first + second + deep_out + self.global_bias}
