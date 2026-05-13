from __future__ import annotations

"""FeatureEmbeddings：离散特征索引 → 稠密嵌入拼接。"""
import torch.nn as nn


class FeatureEmbeddings(nn.Module):
    """Discrete feature indices -> dense embedding concat."""

    def __init__(self, features: list[tuple[str, int, int]]):
        super().__init__()
        self.ordered_names = []
        self.feature_to_idx = {}
        total_dim = 0
        for i, (name, vocab_size, embed_dim) in enumerate(features):
            self.feature_to_idx[name] = i
            self.ordered_names.append(name)
            setattr(self, f"emb_{name}", nn.Embedding(vocab_size, embed_dim))
            total_dim += embed_dim
        self.num_features = len(features)
        self.total_dim = total_dim

    def forward(self, x_inputs):
        embeds = [getattr(self, f"emb_{n}")(x_inputs[n]) for n in self.ordered_names]
        return __import__("torch").cat(embeds, dim=1)

    def forward_stacked(self, x_inputs):
        return [getattr(self, f"emb_{n}")(x_inputs[n]).unsqueeze(1) for n in self.ordered_names]
