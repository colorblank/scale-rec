from __future__ import annotations

"""FeatureEmbeddings：离散特征索引 → 稠密嵌入拼接，支持变长序列池化。"""
import torch
import torch.nn as nn


class FeatureEmbeddings(nn.Module):
    """特征嵌入层，支持 mean/sum/max pooling 处理变长序列特征。

    Args:
        features: [(name, vocab_size, embed_dim), ...]
        pooling_map: {name: "flatten"|"mean"|"sum"|"max"}
    """

    def __init__(
        self,
        features: list[tuple[str, int, int]],
        pooling_map: dict[str, str] | None = None,
    ):
        super().__init__()
        self.ordered_names = []
        self.feature_to_idx = {}
        self.pooling_map = pooling_map or {}
        total_dim = 0
        for i, (name, vocab_size, embed_dim) in enumerate(features):
            self.feature_to_idx[name] = i
            self.ordered_names.append(name)
            setattr(self, f"emb_{name}", nn.Embedding(vocab_size, embed_dim))
            total_dim += embed_dim
        self.num_features = len(features)
        self.total_dim = total_dim

    def _pool(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        p = self.pooling_map.get(name, "flatten")
        if p == "mean":
            return emb.mean(dim=1)
        elif p == "sum":
            return emb.sum(dim=1)
        elif p == "max":
            return emb.max(dim=1).values
        return emb  # flatten: already 2D, no-op

    def forward(self, x_inputs):
        """Lookup + pool + concat → [batch, total_dim]."""
        embeds = []
        for n in self.ordered_names:
            idx = x_inputs[n]  # (batch,) or (batch, seq)
            emb = getattr(self, f"emb_{n}")(idx)  # (batch, d) or (batch, seq, d)
            if idx.dim() == 2:
                emb = self._pool(emb, n)
            embeds.append(emb)
        return torch.cat(embeds, dim=1)

    def forward_stacked(self, x_inputs):
        """Return list of [batch, 1, embed_dim] for FM interaction."""
        result = []
        for n in self.ordered_names:
            idx = x_inputs[n]
            emb = getattr(self, f"emb_{n}")(idx)
            if idx.dim() == 2:
                emb = self._pool(emb, n)
            result.append(emb.unsqueeze(1))
        return result
