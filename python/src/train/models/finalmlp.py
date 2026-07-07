"""FinalMLP: Two-stream MLP with feature gating and interaction aggregation (arXiv:2304.00902)."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation
from .output_head import OutputHead


class FinalMLP(nn.Module):
    """FinalMLP: two parallel MLPs with feature gating + interaction aggregation."""

    def __init__(
        self,
        features: list[FeatureTuple],
        stream_hidden_dims: list[int] | None = None,
        gate_hidden_dim: int = 64,
        fusion_hidden_dims: list[int] | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        stream_hidden_dims = stream_hidden_dims or []
        fusion_hidden_dims = fusion_hidden_dims or []

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        total_dim = self.embeddings.total_dim
        stream_output_dim = stream_hidden_dims[-1] if stream_hidden_dims else 1

        # Separate Linear modules for Rust-compatible weight naming (stream{x}_gate.{0,1})
        self.stream1_gate_0 = nn.Linear(total_dim, gate_hidden_dim)
        self.stream1_gate_1 = nn.Linear(gate_hidden_dim, total_dim)
        self.stream2_gate_0 = nn.Linear(total_dim, gate_hidden_dim)
        self.stream2_gate_1 = nn.Linear(gate_hidden_dim, total_dim)

        self.stream1_mlp = Mlp(
            total_dim,
            stream_hidden_dims[:-1] if stream_hidden_dims else [],
            stream_output_dim,
            Activation.RELU,
        )
        self.stream2_mlp = Mlp(
            total_dim,
            stream_hidden_dims[:-1] if stream_hidden_dims else [],
            stream_output_dim,
            Activation.RELU,
        )

        fusion_input_dim = stream_output_dim * 3
        self.fusion = Mlp(
            fusion_input_dim,
            fusion_hidden_dims,
            1,
            Activation.RELU,
        )

        self.output_head = (
            OutputHead(output_contract, {"shared": 1}) if output_contract is not None else None
        )

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
        x = self.embeddings(x_inputs)

        gate1 = torch.sigmoid(self.stream1_gate_1(torch.relu(self.stream1_gate_0(x))))
        x1 = x * gate1
        h1 = self.stream1_mlp(x1)

        gate2 = torch.sigmoid(self.stream2_gate_1(torch.relu(self.stream2_gate_0(x))))
        x2 = x * gate2
        h2 = self.stream2_mlp(x2)

        combined = torch.cat([h1, h2, h1 * h2], dim=1)
        return self.fusion(combined)
