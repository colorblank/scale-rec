from __future__ import annotations

"""HyFormer: query decoding plus RankMixer-style query boosting."""

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
class HyFormerConfig:
    d: int = 64
    d_ff: int = 128
    num_queries: int = 2
    num_layers: int = 2
    hidden_factor: float = 1.0


class HyFormerModel(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        config: HyFormerConfig,
        task_config: MultiTaskConfig | None,
        output_contract: NormalizedOutputContract | None = None,
        pooling_map: dict[str, str] | None = None,
        seq_len_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = HyFormerTokenizer(features, config.d, pooling_map, seq_len_map)
        self.query_generator = QueryGenerator(
            self.tokenizer.global_input_dim,
            config.d,
            config.d_ff,
            config.num_queries,
        )
        token_count = self.tokenizer.boost_token_count(config.num_queries)
        if config.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if config.hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        self.layers = nn.ModuleList(
            HyFormerLayer(config.d, token_count, config.num_queries, config.hidden_factor)
            for _ in range(config.num_layers)
        )
        self.output_proj = nn.Linear(config.d, config.d)
        if output_contract is not None:
            self.output_head = OutputHead(output_contract, {"shared": config.d})
            self.task_towers = None
        else:
            if task_config is None:
                raise ValueError("HyFormer requires task_config or output_contract")
            self.task_towers = MultiTaskTower(task_config, config.d)

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_head"):
            return self.forward_execution(x_inputs).outputs
        return self.task_towers(self._shared(x_inputs))

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        shared = self._shared(x_inputs)
        if hasattr(self, "output_head"):
            return self.output_head({"shared": shared})
        outputs = self.task_towers(shared)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        encoded = self.tokenizer(x_inputs)
        queries = self.query_generator(encoded.global_info)
        for layer in self.layers:
            queries = layer(queries, encoded.memory, encoded.non_sequence_tokens)
        return self.output_proj(queries.mean(dim=1))


@dataclass
class EncodedFeatures:
    global_info: torch.Tensor
    non_sequence_tokens: torch.Tensor
    memory: torch.Tensor


class HyFormerTokenizer(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        d: int,
        pooling_map: dict[str, str] | None,
        seq_len_map: dict[str, int] | None,
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError("HyFormer requires at least one feature")
        if d <= 0:
            raise ValueError("d must be > 0")
        self.ordered_names = [name for name, _, _ in features]
        self.pooling_map = pooling_map or {}
        self.seq_len_map = seq_len_map or {}
        self.d = d
        self.global_input_dim = 0
        self.non_sequence_count = 0
        self.embeddings = nn.ModuleDict()
        self.pooled_projections = nn.ModuleList()
        self.sequence_projections = nn.ModuleList()
        self.feature_dims: dict[str, int] = {}
        for name, vocab_size, embed_dim in features:
            self.embeddings[f"emb_{name}"] = nn.Embedding(vocab_size, embed_dim)
            pooled_dim = self._pooled_dim(name, embed_dim)
            self.feature_dims[name] = embed_dim
            self.global_input_dim += pooled_dim
            if name not in self.seq_len_map:
                self.non_sequence_count += 1
            self.pooled_projections.append(nn.Linear(pooled_dim, d))
            self.sequence_projections.append(nn.Linear(embed_dim, d))

    def boost_token_count(self, num_queries: int) -> int:
        return num_queries + max(self.non_sequence_count, 1)

    def forward(self, x_inputs: FeatureTensorMap) -> EncodedFeatures:
        global_parts = []
        ns_tokens = []
        memory_tokens = []
        for idx, name in enumerate(self.ordered_names):
            raw = x_inputs[name]
            emb = self.embeddings[f"emb_{name}"](raw)
            pooled = self._pool(emb, name) if emb.dim() == 3 else emb
            global_parts.append(pooled)
            if name in self.seq_len_map and emb.dim() == 3:
                memory_tokens.append(self.sequence_projections[idx](emb))
            else:
                token = self.pooled_projections[idx](pooled).unsqueeze(1)
                ns_tokens.append(token)
                memory_tokens.append(token)
        if not ns_tokens:
            ns_tokens.append(memory_tokens[0].mean(dim=1, keepdim=True))
        return EncodedFeatures(
            global_info=torch.cat(global_parts, dim=1),
            non_sequence_tokens=torch.cat(ns_tokens, dim=1),
            memory=torch.cat(memory_tokens, dim=1),
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


class QueryGenerator(nn.Module):
    def __init__(self, input_dim: int, d: int, d_ff: int, num_queries: int) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be > 0")
        self.num_queries = num_queries
        self.d = d
        self.up = nn.Linear(input_dim, max(d_ff, 1))
        self.down = nn.Linear(max(d_ff, 1), num_queries * d)

    def forward(self, global_info: torch.Tensor) -> torch.Tensor:
        out = self.down(F.gelu(self.up(global_info)))
        return out.view(out.shape[0], self.num_queries, self.d)


class HyFormerLayer(nn.Module):
    def __init__(
        self,
        d: int,
        token_count: int,
        num_queries: int,
        hidden_factor: float,
    ) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.boost = HyFormerBoostBlock(d, token_count, hidden_factor)
        self.num_queries = num_queries

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        ns_tokens: torch.Tensor,
    ) -> torch.Tensor:
        decoded = self._decode(queries, memory)
        boosted = self.boost(torch.cat([decoded, ns_tokens], dim=1))
        return boosted[:, : self.num_queries, :]

    def _decode(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(self.norm_query(queries))
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        scores = torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        decoded = torch.matmul(torch.softmax(scores, dim=2), v)
        return self.out_proj(decoded) + queries


class HyFormerBoostBlock(nn.Module):
    def __init__(self, d: int, token_count: int, hidden_factor: float) -> None:
        super().__init__()
        if token_count <= 0:
            raise ValueError("token_count must be > 0")
        hidden_dim = max(1, int(d * hidden_factor))
        self.norm_mixing = nn.LayerNorm(d)
        self.token_mixing = nn.Linear(d, d)
        self.ffn_up = nn.Linear(d, hidden_dim)
        self.ffn_down = nn.Linear(hidden_dim, d)
        self.norm_ffn = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x.mean(dim=1, keepdim=True)
        mixed = self.token_mixing(context).expand_as(x)
        s = self.norm_mixing(mixed + x)
        return self.norm_ffn(self.ffn_down(F.gelu(self.ffn_up(s))) + s)
