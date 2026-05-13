"""UniMixer：Full model。"""
import torch, torch.nn as nn
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from .block import UniMixerBlock
from .norm import SiameseNorm
from .tokenizer import FeatureTokenizer


class UniMixerModel(nn.Module):
    def __init__(
        self,
        tokenizer,
        token_dim,
        num_tokens,
        num_blocks,
        block_size_opt,
        use_lite,
        hidden_factor,
        num_basis,
        rank,
        task_config,
        use_siamese,
    ):
        super().__init__()
        self.embed_dim = num_tokens * token_dim
        self.block_size = block_size_opt if block_size_opt is not None else token_dim
        self.use_siamese = use_siamese
        self.temperature = 1.0
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
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
                )
            )
        self.task_towers = MultiTaskTower(task_config, self.embed_dim)
        self.final_norm = SiameseNorm(self.embed_dim) if use_siamese else None

    def forward(self, x_inputs, temperature=None):
        t = temperature if temperature is not None else self.temperature
        tokens = self.tokenizer(x_inputs)
        bs = tokens.shape[0]
        x = tokens.reshape(bs, self.embed_dim)
        if self.use_siamese:
            x_bar = y_bar = x
            for blk in self.blocks:
                _, xbn, ybn = blk(x, t, x_bar, y_bar, use_siamese=True)
                x_bar = xbn
                y_bar = ybn
                x = x_bar
            output = self.final_norm(x_bar, y_bar, None)
        else:
            for blk in self.blocks:
                x = blk(x, t, use_siamese=False)
            output = x
        return self.task_towers(output)
