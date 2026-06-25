"""TokenMixer-Large: FeatureTokenizer + M TokenMixerLargeBlocks + MultiTaskTower."""

import torch.nn as nn

from ...core.model_output import ModelExecution, ModelOutput
from ...core.output_contract import NormalizedOutputContract
from ...layers.embedding import FeatureTensorMap
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from ..output_head import OutputHead
from ..unimixer.tokenizer import FeatureTokenizer
from .block import TokenMixerLargeBlock


class TokenMixerLargeModel(nn.Module):
    """Full TokenMixer-Large: FeatureTokenizer + M blocks + task towers."""

    def __init__(
        self,
        tokenizer: FeatureTokenizer,
        token_dim: int,
        num_tokens: int,
        num_blocks: int,
        num_heads: int,
        hidden_factor: float,
        task_config: MultiTaskConfig | None,
        down_init_scale: float = 0.01,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be > 0")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if hidden_factor <= 0:
            raise ValueError("hidden_factor must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        self.embed_dim = num_tokens * token_dim
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TokenMixerLargeBlock(
                    embed_dim=self.embed_dim,
                    token_dim=token_dim,
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    hidden_factor=hidden_factor,
                    down_init_scale=down_init_scale,
                )
            )
        if output_contract is not None:
            self.output_head = OutputHead(output_contract, {"shared": self.embed_dim})
            self.task_towers = None
        else:
            if task_config is None:
                raise ValueError("TokenMixerLarge requires task_config or output_contract")
            self.task_towers = MultiTaskTower(task_config, self.embed_dim)

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

    def _shared(self, x_inputs: FeatureTensorMap):
        tokens = self.tokenizer(x_inputs)
        bs = tokens.shape[0]
        x = tokens.reshape(bs, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        return x
