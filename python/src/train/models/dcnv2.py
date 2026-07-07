"""DCN V2: Improved Deep & Cross Network (arXiv:2008.13535).

Single-task CTR model with gated cross network + optional deep MLP.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.gdcn import GatedCrossNetwork
from ..layers.mlp import Mlp
from ..layers.towers import Activation
from .output_head import OutputHead


class DCNV2(nn.Module):
    """DCN V2: gated cross network + optional deep MLP."""

    def __init__(
        self,
        features: list[FeatureTuple],
        cross_layers: int = 3,
        deep_hidden_dims: list[int] | None = None,
        shared_bottom_dims: list[int] | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        deep_hidden_dims = deep_hidden_dims or []
        shared_bottom_dims = shared_bottom_dims or []

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        input_dim = self.embeddings.total_dim
        self.cross = GatedCrossNetwork(input_dim, cross_layers)

        if deep_hidden_dims:
            self.deep = Mlp(input_dim, deep_hidden_dims[:-1], deep_hidden_dims[-1], Activation.RELU)
            fusion_dim = input_dim + deep_hidden_dims[-1]
        else:
            fusion_dim = input_dim

        if shared_bottom_dims:
            self.shared_bottom = Mlp(
                fusion_dim, shared_bottom_dims[:-1], shared_bottom_dims[-1], Activation.RELU
            )
            shared_dim = shared_bottom_dims[-1]
        else:
            shared_dim = fusion_dim

        self._shared_dim = shared_dim
        self.output_head = (
            OutputHead(output_contract, {"shared": shared_dim})
            if output_contract is not None
            else None
        )

    @property
    def has_deep(self) -> bool:
        return hasattr(self, "deep")

    @property
    def has_shared_bottom(self) -> bool:
        return hasattr(self, "shared_bottom")

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if self.output_head is not None:
            return self.forward_execution(x_inputs).outputs
        return ModelOutput.binary_logits({"pred": self._shared(x_inputs)})

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        if self.output_head is not None:
            return self.output_head({"shared": self._shared(x_inputs)})
        outputs = self.forward(x_inputs)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        dense = self.embeddings(x_inputs)
        cross_out = self.cross(dense)
        shared = torch.cat([cross_out, self.deep(dense)], dim=1) if self.has_deep else cross_out
        if self.has_shared_bottom:
            shared = self.shared_bottom(shared)
        return shared
