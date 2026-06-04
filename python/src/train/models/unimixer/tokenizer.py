from __future__ import annotations

"""FeatureTokenizer：分组 Conv1d 特征投影。"""
from typing import Optional

import torch
import torch.nn as nn

from ...layers.embedding import FeatureTensorMap, FeatureTuple


class FeatureTokenizer(nn.Module):
    """Grouped Conv1d projection: sparse features → pooled → unified token sequence."""

    def __init__(
        self,
        features: list[FeatureTuple],
        token_dim: int,
        num_tokens: int,
        pooling_map: Optional[dict[str, str]] = None,
        seq_len_map: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        self.ordered_names = []
        self.feature_to_idx = {}
        self.pooling_map = pooling_map or {}
        self.seq_len_map = seq_len_map or {}
        total = 0
        for i, (name, vocab_size, embed_dim) in enumerate(features):
            self.feature_to_idx[name] = i
            self.ordered_names.append(name)
            setattr(self, f"emb_{name}", nn.Embedding(vocab_size, embed_dim))
            if self.pooling_map.get(name, "first") == "flatten":
                seq_len = self.seq_len_map.get(name)
                if not seq_len or seq_len <= 0:
                    raise ValueError(f"feature '{name}' pooling flatten requires seq_len > 0")
                total += embed_dim * seq_len
            else:
                total += embed_dim
        if total % num_tokens != 0:
            raise ValueError(
                f"total_embed_dim ({total}) not divisible by num_tokens ({num_tokens})"
            )
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.total_embed_dim = total
        self.token_projections = nn.Conv1d(
            total, num_tokens * token_dim, kernel_size=1, groups=num_tokens
        )

    def _pool(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        p = self.pooling_map.get(name, "first")
        if p == "mean":
            return emb.mean(dim=1)
        elif p == "sum":
            return emb.sum(dim=1)
        elif p == "max":
            return emb.max(dim=1).values
        elif p == "flatten":
            return emb.reshape(emb.shape[0], -1)
        return emb[:, 0, :]

    def forward(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        """Embed → pool(if 2D) → concat → Conv1d → [batch, num_tokens, token_dim]."""
        embeds = []
        for n in self.ordered_names:
            idx = x_inputs[n]
            emb = getattr(self, f"emb_{n}")(idx)
            if idx.dim() == 2:
                emb = self._pool(emb, n)
            embeds.append(emb)
        concat = torch.cat(embeds, dim=1)
        conv_in = concat.unsqueeze(2)
        conv_out = self.token_projections(conv_in)
        squeezed = conv_out.squeeze(2)
        return squeezed.view(squeezed.shape[0], self.num_tokens, self.token_dim)
