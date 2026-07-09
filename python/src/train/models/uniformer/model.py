from __future__ import annotations

"""UniFormer: feature interaction module plus task interaction module."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.model_output import ModelExecution, ModelOutput
from ...core.output_contract import NormalizedOutputContract
from ...layers.embedding import FeatureTensorMap, FeatureTuple
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from ..output_head import OutputHead


@dataclass
class UniFormerConfig:
    d: int = 128
    d_ff: int = 512
    num_layers: int = 2
    n_heads: int = 8
    num_tasks: int = 3


def _rms_norm(d: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d, eps=1e-5)
    return nn.LayerNorm(d, eps=1e-5)


class UniFormerModel(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        config: UniFormerConfig,
        task_config: MultiTaskConfig | None,
        output_contract: NormalizedOutputContract | None = None,
        pooling_map: dict[str, str] | None = None,
        seq_len_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if config.d <= 0:
            raise ValueError("d must be > 0")
        if config.d_ff <= 0:
            raise ValueError("d_ff must be > 0")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if config.n_heads <= 0 or config.d % config.n_heads != 0:
            raise ValueError(f"d ({config.d}) must be divisible by n_heads ({config.n_heads})")
        if config.num_tasks <= 0:
            raise ValueError("num_tasks must be > 0")
        self.tokenizer = UniFormerTokenizer(features, config.d, pooling_map, seq_len_map)
        self.fim_layers = nn.ModuleList(
            FeatureInteractionLayer(config.d, config.d_ff, config.n_heads)
            for _ in range(config.num_layers)
        )
        self.tim_layers = nn.ModuleList(
            TaskInteractionLayer(config.d, config.d_ff, config.n_heads)
            for _ in range(config.num_layers)
        )
        self.task_tokens = nn.Parameter(torch.empty(config.num_tasks, config.d))
        nn.init.kaiming_normal_(self.task_tokens)
        self.final_norm = _rms_norm(config.d)
        self.d = config.d
        self.num_tasks = config.num_tasks
        if output_contract is not None:
            dims = {"shared": config.d}
            dims.update({f"task_{idx}": config.d for idx in range(config.num_tasks)})
            self.output_head = OutputHead(output_contract, dims)
            self.task_towers = None
        else:
            if task_config is None:
                raise ValueError("UniFormer requires task_config or output_contract")
            self.task_towers = MultiTaskTower(task_config, config.d)

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_head"):
            return self.forward_execution(x_inputs).outputs
        return self.task_towers(self._representations(x_inputs)["shared"])

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        reps = self._representations(x_inputs)
        if hasattr(self, "output_head"):
            return self.output_head(reps)
        outputs = self.task_towers(reps["shared"])
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _representations(self, x_inputs: FeatureTensorMap) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(x_inputs)
        feature_tokens = encoded.non_sequence_tokens
        sequence_tokens = encoded.sequence_tokens
        for layer in self.fim_layers:
            feature_tokens = layer(feature_tokens, sequence_tokens)
        task_tokens = self.task_tokens.unsqueeze(0).expand(feature_tokens.shape[0], -1, -1)
        for layer in self.tim_layers:
            task_tokens = layer(task_tokens, feature_tokens)
        task_tokens = self.final_norm(task_tokens)
        reps = {"shared": task_tokens.mean(dim=1)}
        reps.update({f"task_{idx}": task_tokens[:, idx, :] for idx in range(self.num_tasks)})
        return reps


@dataclass
class EncodedFeatures:
    non_sequence_tokens: torch.Tensor
    sequence_tokens: torch.Tensor


class UniFormerTokenizer(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        d: int,
        pooling_map: dict[str, str] | None,
        seq_len_map: dict[str, int] | None,
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError("UniFormer requires at least one feature")
        self.ordered_names = [name for name, _, _ in features]
        self.pooling_map = pooling_map or {}
        self.seq_len_map = seq_len_map or {}
        self.embeddings = nn.ModuleDict()
        self.sequence_projections = nn.ModuleList()
        self.non_sequence_projections = nn.ModuleList()
        for name, vocab_size, embed_dim in features:
            self.embeddings[f"emb_{name}"] = nn.Embedding(vocab_size, embed_dim)
            self.sequence_projections.append(nn.Linear(embed_dim, d))
            self.non_sequence_projections.append(nn.Linear(self._pooled_dim(name, embed_dim), d))

    def forward(self, x_inputs: FeatureTensorMap) -> EncodedFeatures:
        non_sequence_tokens = []
        sequence_tokens = []
        for idx, name in enumerate(self.ordered_names):
            raw = x_inputs[name]
            emb = self.embeddings[f"emb_{name}"](raw)
            if name in self.seq_len_map and emb.dim() == 3:
                sequence_tokens.append(self.sequence_projections[idx](emb))
            else:
                pooled = self._pool(emb, name) if emb.dim() == 3 else emb
                non_sequence_tokens.append(self.non_sequence_projections[idx](pooled).unsqueeze(1))
        if not sequence_tokens:
            sequence_tokens.extend(non_sequence_tokens)
        if not non_sequence_tokens:
            non_sequence_tokens.append(torch.cat(sequence_tokens, dim=1).mean(dim=1, keepdim=True))
        return EncodedFeatures(
            non_sequence_tokens=torch.cat(non_sequence_tokens, dim=1),
            sequence_tokens=torch.cat(sequence_tokens, dim=1),
        )

    def _pooled_dim(self, name: str, embed_dim: int) -> int:
        if self.pooling_map.get(name, "first") == "flatten":
            seq_len = self.seq_len_map.get(name)
            if not seq_len or seq_len <= 0:
                raise ValueError(f"feature '{name}' pooling flatten requires seq_len > 0")
            return embed_dim * seq_len
        return embed_dim

    def _pool(self, emb: torch.Tensor, name: str) -> torch.Tensor:
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


class FeatureInteractionLayer(nn.Module):
    def __init__(self, d: int, d_ff: int, n_heads: int) -> None:
        super().__init__()
        self.cross_attn = MultiHeadAttention(d, n_heads)
        self.self_attn = MultiHeadAttention(d, n_heads)
        self.ffn = SwiGluFfn(d, d_ff)
        self.norm_cross = _rms_norm(d)
        self.norm_self = _rms_norm(d)
        self.norm_ffn = _rms_norm(d)

    def forward(self, non_sequence: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        x = non_sequence + self.cross_attn(self.norm_cross(non_sequence), sequence, sequence)
        y = x + self.self_attn(self.norm_self(x), x, x)
        return y + self.ffn(self.norm_ffn(y))


class TaskInteractionLayer(nn.Module):
    def __init__(self, d: int, d_ff: int, n_heads: int) -> None:
        super().__init__()
        self.cross_attn = MultiHeadAttention(d, n_heads)
        self.self_attn = MultiHeadAttention(d, n_heads)
        self.ffn = SwiGluFfn(d, d_ff)
        self.norm_cross = _rms_norm(d)
        self.norm_self = _rms_norm(d)
        self.norm_ffn = _rms_norm(d)

    def forward(self, task_tokens: torch.Tensor, feature_tokens: torch.Tensor) -> torch.Tensor:
        x = task_tokens + self.cross_attn(
            self.norm_cross(task_tokens), feature_tokens, feature_tokens
        )
        y = x + self.self_attn(self.norm_self(x), x, x)
        return y + self.ffn(self.norm_ffn(y))


class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        k_len = key.shape[1]
        q = self.q_proj(query).view(batch, q_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch, k_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch, k_len, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head**0.5)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.d)
        return self.o_proj(out)


class SwiGluFfn(nn.Module):
    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.up = nn.Linear(d, d_ff)
        self.gate = nn.Linear(d, d_ff)
        self.down = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)) * torch.sigmoid(self.gate(x)))
