"""RankMixer: FeatureTokenizer + dense RankMixer blocks + MultiTaskTower."""

import torch
import torch.nn as nn

from ...layers.embedding import FeatureTensorMap
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from ..unimixer.tokenizer import FeatureTokenizer
from .block import RankMixerBlock


class RankMixerModel(nn.Module):
    """Dense RankMixer model with mean-pooled token output."""

    def __init__(
        self,
        tokenizer: FeatureTokenizer,
        token_dim: int,
        num_tokens: int,
        num_blocks: int,
        num_heads: int,
        hidden_factor: float,
        task_config: MultiTaskConfig,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList(
            RankMixerBlock(token_dim, num_tokens, num_heads, hidden_factor)
            for _ in range(num_blocks)
        )
        self.task_towers = MultiTaskTower(task_config, token_dim)

    def forward(self, x_inputs: FeatureTensorMap) -> dict[str, torch.Tensor]:
        x = self.tokenizer(x_inputs)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.task_towers(pooled)
