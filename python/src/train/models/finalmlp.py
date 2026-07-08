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
    """FinalMLP: two parallel gated MLPs with learnable bilinear aggregation."""

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
        _ = fusion_hidden_dims or []

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        total_dim = self.embeddings.total_dim
        stream_output_dim = stream_hidden_dims[-1] if stream_hidden_dims else 1

        self.stream1_gate = nn.Sequential(
            nn.Linear(total_dim, gate_hidden_dim),
            nn.Linear(gate_hidden_dim, total_dim),
        )
        self.stream2_gate = nn.Sequential(
            nn.Linear(total_dim, gate_hidden_dim),
            nn.Linear(gate_hidden_dim, total_dim),
        )

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

        self.fusion_o1 = nn.Linear(stream_output_dim, 1, bias=False)
        self.fusion_o2 = nn.Linear(stream_output_dim, 1, bias=False)
        self.fusion_bilinear = nn.Linear(stream_output_dim, stream_output_dim, bias=False)
        self.fusion_bias = nn.Parameter(torch.zeros(1))

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

    def _gate(self, x: torch.Tensor, gate: nn.Sequential) -> torch.Tensor:
        return 2.0 * torch.sigmoid(gate[1](torch.relu(gate[0](x))))

    def _bilinear_fusion(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        linear_terms = self.fusion_o1(h1) + self.fusion_o2(h2)
        bilinear = (self.fusion_bilinear(h1) * h2).sum(dim=1, keepdim=True)
        return linear_terms + bilinear + self.fusion_bias

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        x = self.embeddings(x_inputs)

        x1 = x * self._gate(x, self.stream1_gate)
        h1 = self.stream1_mlp(x1)

        x2 = x * self._gate(x, self.stream2_gate)
        h2 = self.stream2_mlp(x2)

        return self._bilinear_fusion(h1, h2)
