from __future__ import annotations

"""DeepFM：FM 一阶 + FM 二阶 + Deep MLP。"""

import torch
import torch.nn as nn

from ..core.config import PoolingMode
from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.fm import fm_interaction
from ..layers.mlp import Mlp
from ..layers.towers import Activation
from .output_head import OutputHead


class DeepFM(nn.Module):
    """DeepFM: FM first-order + FM second-order + Deep MLP."""

    def __init__(
        self,
        features: list[FeatureTuple],
        fm_k: int,
        deep_hidden_dims: list[int],
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        self.fm_first = FeatureEmbeddings([(n, v, 1) for n, v, _ in features], pooling_map)
        self.fm_second = FeatureEmbeddings([(n, v, fm_k) for n, v, _ in features], pooling_map)
        self.fm_k = fm_k
        self.deep = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        self.deep_total_dim = self.deep.total_dim
        self.deep_mlp = Mlp(self.deep_total_dim, deep_hidden_dims, 1, Activation.RELU)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.output_head = (
            OutputHead(output_contract, {"shared": 1}) if output_contract is not None else None
        )

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        """Forward: FM first + FM second + Deep MLP + global_bias -> {"pred": logits}."""
        if self.output_head is not None:
            return self.forward_execution(x_inputs).outputs
        return ModelOutput.binary_logits({"pred": self._shared(x_inputs)})

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        if self.output_head is not None:
            return self.output_head({"shared": self._shared(x_inputs)})
        outputs = self.forward(x_inputs)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        first = self.fm_first(x_inputs).sum(dim=1, keepdim=True)
        stacked = torch.cat(self.fm_second.forward_stacked(x_inputs), dim=1)
        second = fm_interaction(stacked)
        deep_out = self.deep_mlp(self.deep(x_inputs))
        return first + second + deep_out + self.global_bias
