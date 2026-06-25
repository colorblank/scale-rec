from __future__ import annotations

"""逻辑回归基线：Embedding + Linear。"""

from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation
from .output_head import OutputHead


class LogisticRegression(nn.Module):
    """LR baseline: Embedding + Linear, no feature interaction."""

    def __init__(
        self,
        features: list[FeatureTuple],
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        self.mlp = Mlp(self.embeddings.total_dim, [], 1, Activation.NONE)
        self.output_head = (
            OutputHead(output_contract, {"shared": 1}) if output_contract is not None else None
        )

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        """Forward: embed -> concat -> linear -> {"pred": logits}."""
        if self.output_head is not None:
            return self.forward_execution(x_inputs).outputs
        return ModelOutput.binary_logits({"pred": self._shared(x_inputs)})

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        if self.output_head is not None:
            return self.output_head({"shared": self._shared(x_inputs)})
        outputs = self.forward(x_inputs)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap):
        return self.mlp(self.embeddings(x_inputs))
