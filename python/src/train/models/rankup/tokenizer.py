from __future__ import annotations

"""RankUp tokenizer: shuffled sparse groups plus optional auxiliary tokens."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...layers.embedding import FeatureTensorMap, FeatureTuple


@dataclass(frozen=True)
class CrossTokenConfig:
    left: str
    right: str


class RankUpTokenizer(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        token_dim: int,
        num_sparse_tokens: int,
        permutation_seed: int,
        multi_embedding_tables: int,
        use_global_token: bool,
        cross_token: CrossTokenConfig | None,
        num_task_tokens: int,
        pooling_map: dict[str, str] | None = None,
        seq_len_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError("RankUp requires at least one feature")
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if num_sparse_tokens <= 0:
            raise ValueError("num_sparse_tokens must be > 0")
        if num_sparse_tokens > len(features):
            raise ValueError(
                f"num_sparse_tokens ({num_sparse_tokens}) cannot exceed feature count ({len(features)})"
            )
        if multi_embedding_tables <= 0:
            raise ValueError("multi_embedding_tables must be > 0")

        self.ordered_names = [name for name, _, _ in features]
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.ordered_names)}
        self.pooling_map = pooling_map or {}
        self.seq_len_map = seq_len_map or {}
        self.token_dim = token_dim
        self.sparse_groups = _shuffled_groups(len(features), num_sparse_tokens, permutation_seed)

        feature_dims: list[int] = []
        for name, vocab_size, embed_dim in features:
            setattr(self, f"emb_{name}", nn.Embedding(vocab_size, embed_dim))
            feature_dims.append(self._feature_output_dim(name, embed_dim) * multi_embedding_tables)
        for table_idx in range(1, multi_embedding_tables):
            for name, vocab_size, embed_dim in features:
                setattr(self, f"multi_emb_{table_idx}_{name}", nn.Embedding(vocab_size, embed_dim))

        self.token_projections = nn.ModuleList(
            nn.Linear(sum(feature_dims[idx] for idx in group), token_dim)
            for group in self.sparse_groups
        )
        self.global_projection = (
            nn.Linear(sum(feature_dims), token_dim) if use_global_token else None
        )

        self.cross_pair: tuple[int, int] | None = None
        self.cross_projection: nn.Linear | None = None
        if cross_token is not None:
            if cross_token.left not in self.feature_to_idx:
                raise ValueError(f"cross token left feature '{cross_token.left}' not found")
            if cross_token.right not in self.feature_to_idx:
                raise ValueError(f"cross token right feature '{cross_token.right}' not found")
            left = self.feature_to_idx[cross_token.left]
            right = self.feature_to_idx[cross_token.right]
            left_dim = self._feature_output_dim(features[left][0], features[left][2])
            right_dim = self._feature_output_dim(features[right][0], features[right][2])
            if left_dim != right_dim:
                raise ValueError(
                    f"cross token features '{cross_token.left}' and '{cross_token.right}' "
                    f"must have equal pooled dims, got {left_dim} and {right_dim}"
                )
            self.cross_pair = (left, right)
            self.cross_projection = nn.Linear(left_dim, token_dim)

        self.task_tokens = (
            nn.Parameter(torch.empty(num_task_tokens, token_dim).normal_(0.0, 0.02))
            if num_task_tokens > 0
            else None
        )
        self.num_tokens = (
            num_sparse_tokens
            + int(use_global_token)
            + int(self.cross_pair is not None)
            + num_task_tokens
        )

    @property
    def num_task_tokens(self) -> int:
        return 0 if self.task_tokens is None else self.task_tokens.shape[0]

    def _feature_output_dim(self, name: str, embed_dim: int) -> int:
        if self.pooling_map.get(name, "first") == "flatten":
            seq_len = self.seq_len_map.get(name)
            if not seq_len or seq_len <= 0:
                raise ValueError(f"feature '{name}' pooling flatten requires seq_len > 0")
            return embed_dim * seq_len
        return embed_dim

    def _pool(self, emb: torch.Tensor, name: str) -> torch.Tensor:
        if emb.dim() != 3:
            return emb
        mode = self.pooling_map.get(name, "first")
        if mode == "mean":
            return emb.mean(dim=1)
        if mode == "sum":
            return emb.sum(dim=1)
        if mode == "max":
            return emb.max(dim=1).values
        if mode == "flatten":
            return emb.reshape(emb.shape[0], -1)
        return emb[:, 0, :]

    def forward(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        feature_embeds = []
        for name in self.ordered_names:
            idx = x_inputs[name]
            parts = [self._pool(getattr(self, f"emb_{name}")(idx), name)]
            table_idx = 1
            while hasattr(self, f"multi_emb_{table_idx}_{name}"):
                parts.append(self._pool(getattr(self, f"multi_emb_{table_idx}_{name}")(idx), name))
                table_idx += 1
            feature_embeds.append(torch.cat(parts, dim=1))

        tokens = []
        for group, projection in zip(self.sparse_groups, self.token_projections, strict=True):
            grouped = torch.cat([feature_embeds[idx] for idx in group], dim=1)
            tokens.append(projection(grouped).unsqueeze(1))
        if self.global_projection is not None:
            tokens.append(self.global_projection(torch.cat(feature_embeds, dim=1)).unsqueeze(1))
        if self.cross_pair is not None and self.cross_projection is not None:
            left, right = self.cross_pair
            tokens.append(
                self.cross_projection(feature_embeds[left] * feature_embeds[right]).unsqueeze(1)
            )
        if self.task_tokens is not None:
            tokens.append(self.task_tokens.unsqueeze(0).expand(feature_embeds[0].shape[0], -1, -1))
        return torch.cat(tokens, dim=1)


def _shuffled_groups(count: int, groups: int, seed: int) -> list[list[int]]:
    indices = list(range(count))
    state = max(seed, 1)
    for i in range(len(indices) - 1, 0, -1):
        state = (state * 6364136223846793005 + 1) & ((1 << 64) - 1)
        j = state % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    out = [[] for _ in range(groups)]
    for i, idx in enumerate(indices):
        out[i % groups].append(idx)
    return out
