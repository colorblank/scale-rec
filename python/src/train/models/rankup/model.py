from __future__ import annotations

"""RankUp model."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...core.model_output import ModelExecution, ModelOutput
from ...core.output_contract import NormalizedOutputContract
from ...layers.embedding import FeatureTensorMap, FeatureTuple
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from ..output_head import OutputHead
from ..rankmixer.block import RankMixerBlock
from .tokenizer import CrossTokenConfig, RankUpTokenizer


@dataclass
class RankUpConfig:
    token_dim: int = 64
    num_sparse_tokens: int = 4
    num_blocks: int = 2
    num_heads: int | None = None
    hidden_factor: float = 1.0
    permutation_seed: int = 2026
    multi_embedding_tables: int = 1
    use_global_token: bool = True
    cross_token: CrossTokenConfig | None = None
    num_task_tokens: int = 0


class RankUpModel(nn.Module):
    def __init__(
        self,
        features: list[FeatureTuple],
        config: RankUpConfig,
        task_config: MultiTaskConfig | None,
        output_contract: NormalizedOutputContract | None = None,
        pooling_map: dict[str, str] | None = None,
        seq_len_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if output_contract is not None and config.num_task_tokens == 0:
            config = RankUpConfig(**{**config.__dict__, "num_task_tokens": len(output_contract.towers)})
        self.tokenizer = RankUpTokenizer(
            features,
            config.token_dim,
            config.num_sparse_tokens,
            config.permutation_seed,
            config.multi_embedding_tables,
            config.use_global_token,
            config.cross_token,
            config.num_task_tokens,
            pooling_map,
            seq_len_map,
        )
        if config.num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if config.hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        num_heads = config.num_heads or self.tokenizer.num_tokens
        self.blocks = nn.ModuleList(
            RankMixerBlock(config.token_dim, self.tokenizer.num_tokens, num_heads, config.hidden_factor)
            for _ in range(config.num_blocks)
        )
        self.token_dim = config.token_dim
        if output_contract is not None:
            representation_dims = {"shared": config.token_dim}
            for idx in range(self.tokenizer.num_task_tokens):
                representation_dims[f"task_{idx}"] = config.token_dim * 2
            self.output_head = OutputHead(output_contract, representation_dims)
            self.task_towers = None
        else:
            if task_config is None:
                raise ValueError("RankUp requires task_config or output_contract")
            self.task_towers = MultiTaskTower(task_config, config.token_dim)

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_head"):
            return self.forward_execution(x_inputs).outputs
        return self.task_towers(self._shared(x_inputs))

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        if hasattr(self, "output_head"):
            return self.output_head(self._representations(x_inputs))
        outputs = self.task_towers(self._shared(x_inputs))
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _encoded(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        x = self.tokenizer(x_inputs)
        for block in self.blocks:
            x = block(x)
        return x

    def _representations(self, x_inputs: FeatureTensorMap) -> dict[str, torch.Tensor]:
        x = self._encoded(x_inputs)
        task_tokens = self.tokenizer.num_task_tokens
        shared_tokens = x.shape[1] - task_tokens
        shared = x[:, :shared_tokens, :].mean(dim=1)
        representations = {"shared": shared}
        if task_tokens:
            task_slice = x[:, shared_tokens:, :]
            for idx in range(task_tokens):
                representations[f"task_{idx}"] = torch.cat([task_slice[:, idx, :], shared], dim=1)
        return representations

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        return self._representations(x_inputs)["shared"]
