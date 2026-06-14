from __future__ import annotations

"""UniMixer：Full model。"""

import torch
import torch.nn as nn

from ...layers.embedding import FeatureTensorMap
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from .block import UniMixerBlock
from .norm import SiameseNorm
from .tokenizer import FeatureTokenizer


class UniMixerModel(nn.Module):
    """Full UniMixer: FeatureTokenizer + M blocks + MultiTaskTower + optional SiameseNorm."""

    def __init__(
        self,
        tokenizer: FeatureTokenizer,
        token_dim: int,
        num_tokens: int,
        num_blocks: int,
        block_size_opt: int | None,
        use_lite: bool,
        hidden_factor: float,
        num_basis: int,
        rank: int,
        task_config: MultiTaskConfig,
        use_siamese: bool,
    ) -> None:
        """Build UniMixer: tokenizer + num_blocks UniMixerBlocks + task towers + optional SiameseNorm."""
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        if use_lite and num_basis <= 0:
            raise ValueError("num_basis must be > 0 when use_lite=true")
        if use_lite and rank <= 0:
            raise ValueError("rank must be > 0 when use_lite=true")
        self.embed_dim = num_tokens * token_dim
        self.block_size = block_size_opt if block_size_opt is not None else token_dim
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.embed_dim % self.block_size != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by block_size ({self.block_size})"
            )
        self.use_siamese = use_siamese
        self.temperature = 1.0
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                UniMixerBlock(
                    self.embed_dim,
                    self.block_size,
                    token_dim,
                    num_tokens,
                    use_lite,
                    hidden_factor,
                    num_basis,
                    rank,
                    use_siamese,
                )
            )
        self.task_towers = MultiTaskTower(task_config, self.embed_dim)
        self.final_norm = SiameseNorm(self.embed_dim) if use_siamese else None

    def forward(
        self, x_inputs: FeatureTensorMap, temperature: float | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward: tokenize -> M blocks (standard or siamese path) -> task towers."""
        t = temperature if temperature is not None else self.temperature
        if t <= 0:
            raise ValueError("temperature must be > 0")
        tokens = self.tokenizer(x_inputs)
        bs = tokens.shape[0]
        x = tokens.reshape(bs, self.embed_dim)
        if self.use_siamese:
            x_bar = y_bar = x
            for blk in self.blocks:
                _, xbn, ybn = blk(x, t, x_bar, y_bar)
                x_bar = xbn
                y_bar = ybn
                x = x_bar
            output = self.final_norm(x_bar, y_bar, None)
        else:
            for blk in self.blocks:
                x = blk(x, t)
            output = x
        return self.task_towers(output)
