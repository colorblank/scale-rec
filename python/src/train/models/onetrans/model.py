from __future__ import annotations

"""OneTrans: unified causal transformer for sequence and non-sequence features."""

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
class OneTransConfig:
    d: int = 128
    d_ff: int = 512
    num_layers: int = 2
    n_heads: int = 8
    pyramid_tail_tokens: int | None = None


def _rms_norm(d: int) -> nn.Module:
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d, eps=1e-5)
    return nn.LayerNorm(d, eps=1e-5)


class OneTransModel(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        config: OneTransConfig,
        task_config: MultiTaskConfig | None,
        output_contract: NormalizedOutputContract | None = None,
        pooling_map: dict[str, str] | None = None,
        seq_len_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = OneTransTokenizer(features, config.d, pooling_map, seq_len_map)
        self.blocks = nn.ModuleList(
            OneTransBlock(
                config.d,
                config.d_ff,
                config.n_heads,
                self.tokenizer.max_tokens,
                config.pyramid_tail_tokens,
            )
            for _ in range(config.num_layers)
        )
        self.final_norm = _rms_norm(config.d)
        self.output_proj = nn.Linear(config.d, config.d)
        if output_contract is not None:
            self.output_head = OutputHead(output_contract, {"shared": config.d})
            self.task_towers = None
        else:
            if task_config is None:
                raise ValueError("OneTrans requires task_config or output_contract")
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
        for block in self.blocks:
            encoded = block(encoded)
        pooled = self.final_norm(encoded.tokens).mean(dim=1)
        return self.output_proj(pooled)


@dataclass
class EncodedTokens:
    tokens: torch.Tensor
    is_sequence: list[bool]


class OneTransTokenizer(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        d: int,
        pooling_map: dict[str, str] | None,
        seq_len_map: dict[str, int] | None,
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError("OneTrans requires at least one feature")
        if d <= 0:
            raise ValueError("d must be > 0")
        self.ordered_names = [name for name, _, _ in features]
        self.pooling_map = pooling_map or {}
        self.seq_len_map = seq_len_map or {}
        self.embeddings = nn.ModuleDict()
        self.sequence_projections = nn.ModuleList()
        self.non_sequence_projections = nn.ModuleList()
        self.max_tokens = 0
        for name, vocab_size, embed_dim in features:
            self.embeddings[f"emb_{name}"] = nn.Embedding(vocab_size, embed_dim)
            self.sequence_projections.append(nn.Linear(embed_dim, d))
            self.non_sequence_projections.append(nn.Linear(self._pooled_dim(name, embed_dim), d))
            self.max_tokens += self.seq_len_map.get(name, 1)
        self.position_embedding = nn.Parameter(torch.empty(self.max_tokens, d))
        nn.init.kaiming_normal_(self.position_embedding)

    def forward(self, x_inputs: FeatureTensorMap) -> EncodedTokens:
        sequence_tokens = []
        non_sequence_tokens = []
        is_sequence = []
        for idx, name in enumerate(self.ordered_names):
            raw = x_inputs[name]
            emb = self.embeddings[f"emb_{name}"](raw)
            if name in self.seq_len_map and emb.dim() == 3:
                projected = self.sequence_projections[idx](emb)
                sequence_tokens.append(projected)
                is_sequence.extend([True] * projected.shape[1])
            else:
                pooled = self._pool(emb, name) if emb.dim() == 3 else emb
                token = self.non_sequence_projections[idx](pooled).unsqueeze(1)
                non_sequence_tokens.append(token)
                is_sequence.append(False)
        tokens = torch.cat([*sequence_tokens, *non_sequence_tokens], dim=1)
        tokens = tokens + self.position_embedding[: tokens.shape[1]].unsqueeze(0)
        return EncodedTokens(tokens=tokens, is_sequence=is_sequence)

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


class OneTransBlock(nn.Module):
    def __init__(
        self,
        d: int,
        d_ff: int,
        n_heads: int,
        max_tokens: int,
        pyramid_tail_tokens: int | None,
    ) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError("d must be > 0")
        if d_ff <= 0:
            raise ValueError("d_ff must be > 0")
        if n_heads <= 0 or d % n_heads != 0:
            raise ValueError(f"d ({d}) must be divisible by n_heads ({n_heads})")
        if pyramid_tail_tokens == 0:
            raise ValueError("pyramid_tail_tokens must be > 0 when set")
        self.norm1 = _rms_norm(d)
        self.attn = nn.ModuleDict(
            {
                "q_shared": nn.Linear(d, d),
                "k_shared": nn.Linear(d, d),
                "v_shared": nn.Linear(d, d),
                "q_non_sequence": nn.Linear(d, d),
                "k_non_sequence": nn.Linear(d, d),
                "v_non_sequence": nn.Linear(d, d),
                "o_proj": nn.Linear(d, d),
            }
        )
        self.norm2 = _rms_norm(d)
        self.seq_ffn = nn.ModuleDict(
            {
                "up": nn.Linear(d, d_ff),
                "down": nn.Linear(d_ff, d),
            }
        )
        self.ns_ffn = nn.ModuleDict(
            {
                "up": nn.Linear(d, d_ff),
                "down": nn.Linear(d_ff, d),
            }
        )
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.pyramid_tail_tokens = pyramid_tail_tokens

    def forward(self, encoded: EncodedTokens) -> EncodedTokens:
        x = encoded.tokens
        residual = x
        x = self.norm1(x)
        q = self._project_mixed(
            x, encoded.is_sequence, self.attn["q_shared"], self.attn["q_non_sequence"]
        )
        k = self._project_mixed(
            x, encoded.is_sequence, self.attn["k_shared"], self.attn["k_non_sequence"]
        )
        v = self._project_mixed(
            x, encoded.is_sequence, self.attn["v_shared"], self.attn["v_non_sequence"]
        )
        batch, num_tokens, _ = q.shape
        q = q.view(batch, num_tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(batch, num_tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(batch, num_tokens, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_head**0.5)
        mask = torch.triu(
            torch.full((num_tokens, num_tokens), float("-inf"), device=x.device),
            diagonal=1,
        )
        attn = torch.softmax(scores + mask.view(1, 1, num_tokens, num_tokens), dim=-1)
        x = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(batch, num_tokens, self.d)
        x = residual + self.attn["o_proj"](x)

        residual = x
        x = self.norm2(x)
        x = residual + self._forward_ffn(x, encoded.is_sequence)
        return self._truncate_tail(x, encoded.is_sequence)

    def _project_mixed(
        self,
        x: torch.Tensor,
        is_sequence: list[bool],
        shared: nn.Linear,
        non_sequence: nn.Linear,
    ) -> torch.Tensor:
        outputs = []
        for idx, is_seq in enumerate(is_sequence):
            layer = shared if is_seq else non_sequence
            outputs.append(layer(x[:, idx, :]).unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def _forward_ffn(self, x: torch.Tensor, is_sequence: list[bool]) -> torch.Tensor:
        outputs = []
        for idx, is_seq in enumerate(is_sequence):
            token = x[:, idx, :]
            if is_seq:
                out = self.seq_ffn["down"](F.gelu(self.seq_ffn["up"](token)))
            else:
                out = self.ns_ffn["down"](F.gelu(self.ns_ffn["up"](token)))
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def _truncate_tail(self, tokens: torch.Tensor, is_sequence: list[bool]) -> EncodedTokens:
        keep = self.pyramid_tail_tokens
        if keep is None or tokens.shape[1] <= keep:
            return EncodedTokens(tokens=tokens, is_sequence=list(is_sequence))
        return EncodedTokens(tokens=tokens[:, -keep:, :], is_sequence=is_sequence[-keep:])
