"""Full-Mix: FeatureTokenizer + parameterized full mixing blocks + MultiTaskTower."""

import torch.nn as nn

from ...core.model_output import ModelExecution, ModelOutput
from ...core.output_contract import NormalizedOutputContract
from ...layers.embedding import FeatureTensorMap
from ...layers.towers import MultiTaskConfig, MultiTaskTower
from ..output_head import OutputHead
from ..unimixer.tokenizer import FeatureTokenizer
from .block import FullMixBlock


class FullMixModel(nn.Module):
    """Dense Full-Mix model with mean-pooled token output."""

    def __init__(
        self,
        tokenizer: FeatureTokenizer,
        token_dim: int,
        num_tokens: int,
        num_blocks: int,
        hidden_factor: float,
        task_config: MultiTaskConfig | None,
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
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.tokenizer = tokenizer
        self.blocks = nn.ModuleList(
            FullMixBlock(token_dim, num_tokens, hidden_factor) for _ in range(num_blocks)
        )
        if output_contract is not None:
            self.output_head = OutputHead(output_contract, {"shared": token_dim})
            self.task_towers = None
        else:
            if task_config is None:
                raise ValueError("FullMix requires task_config or output_contract")
            self.task_towers = MultiTaskTower(task_config, token_dim)

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
        x = self.tokenizer(x_inputs)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=1)
