"""PEPNet: Parameter and Embedding Personalized Network.

Architecture:
  FeatureEmbeddings → EPNet gate → Deep MLP → shared_bottom → PPNet gate → towers

EPNet conditions an element-wise gate on the learned prior (aggregated from
all feature embeddings). PPNet applies a similar gate to the shared
representation before task towers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TaskTower
from .esmm import _probability_for_relation
from .output_head import OutputHead


class GateNU(nn.Module):
    """Gate Neural Unit: ReLU hidden layer followed by gamma-scaled sigmoid gate."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, gamma: float = 2.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class PEPNet(nn.Module):
    """PEPNet: personalized embedding and parameter gating for multi-task learning."""

    def __init__(
        self,
        features: list[FeatureTuple],
        prior_dim: int = 16,
        deep_hidden_dims: list[int] | None = None,
        shared_bottom_dims: list[int] | None = None,
        task_config: MultiTaskConfig | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        deep_hidden_dims = deep_hidden_dims or []
        shared_bottom_dims = shared_bottom_dims or []

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        total_dim = self.embeddings.total_dim
        num_features = self.embeddings.num_features

        # Prior projection: num_features → prior_dim
        self.prior_proj = nn.Linear(num_features, prior_dim, bias=False)

        # EPNet gate: prior_dim → total_dim, scaled to (0, 2) as in Gate NU.
        self.epnet_gate = GateNU(prior_dim, prior_dim, total_dim)

        # Deep MLP
        self.has_deep = bool(deep_hidden_dims)
        if self.has_deep:
            self.deep = Mlp(total_dim, deep_hidden_dims[:-1], deep_hidden_dims[-1], Activation.RELU)
            fusion_dim = deep_hidden_dims[-1]
        else:
            fusion_dim = total_dim

        # Shared bottom
        if shared_bottom_dims:
            self.shared_bottom = Mlp(
                fusion_dim, shared_bottom_dims[:-1], shared_bottom_dims[-1], Activation.RELU,
            )
            shared_dim = shared_bottom_dims[-1]
        else:
            shared_dim = fusion_dim

        # PPNet gate: prior_dim → shared_dim, scaled to (0, 2) as in Gate NU.
        self.ppnet_gate = GateNU(prior_dim, prior_dim, shared_dim)

        if output_contract is not None:
            self.output_head = OutputHead(output_contract, {"shared": shared_dim})
            self.task_names = [tower.name for tower in output_contract.towers]
            self.relation_names = list(output_contract.relation_order)
            return

        self.task_config = task_config
        if task_config is None:
            raise ValueError("PEPNet requires task_config or output_contract")
        self.task_names = [tower.name for tower in self.task_config.towers]
        self.relation_names = [relation.target for relation in self.task_config.relations]
        for tower in self.task_config.towers:
            setattr(self, f"{tower.name}_tower", TaskTower(tower, shared_dim))

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_head"):
            return self.forward_execution(x_inputs).outputs
        return self._forward_legacy(self._shared(x_inputs))

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        shared = self._shared(x_inputs)
        if hasattr(self, "output_head"):
            return self.output_head({"shared": shared})
        outputs = self._forward_legacy(shared)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> torch.Tensor:
        stacked = self.embeddings.forward_stacked(x_inputs)
        # stacked: list of [batch, 1, dim_i]

        # Prior: mean-pool each feature → concat → project
        prior_parts = [emb.mean(dim=2, keepdim=True) for emb in stacked]  # [batch, 1, 1]
        prior_raw = torch.cat(prior_parts, dim=1)  # [batch, num_features, 1]
        prior_raw = prior_raw.squeeze(2)           # [batch, num_features]
        prior_raw = prior_raw.detach()
        prior = self.prior_proj(prior_raw)         # [batch, prior_dim]

        # EPNet gate on embeddings
        epnet_scale = self.epnet_gate(prior)  # [batch, total_dim]
        dense_concat = torch.cat([e.squeeze(1) for e in stacked], dim=1)  # [batch, total_dim]
        gated = dense_concat * epnet_scale

        shared = self.deep(gated) if self.has_deep else gated
        if hasattr(self, "shared_bottom"):
            shared = self.shared_bottom(shared)

        # PPNet gate on shared representation
        ppnet_scale = self.ppnet_gate(prior)  # [batch, shared_dim]
        gated_shared = shared * ppnet_scale
        return gated_shared

    def _forward_legacy(self, shared: torch.Tensor) -> ModelOutput:
        outputs = ModelOutput()
        for name in self.task_names:
            tower = getattr(self, f"{name}_tower")
            outputs.insert(name, tower(shared), tower.output_kind)
        for relation in self.task_config.relations:
            outputs.insert_probability(relation.target, self._apply_relation(relation, outputs))
        return outputs

    @staticmethod
    def _apply_relation(relation: TaskRelation, outputs: ModelOutput) -> torch.Tensor:
        if not relation.sources:
            raise ValueError(f"Relation '{relation.target}' has no sources")
        probs = [
            _probability_for_relation(relation, outputs, source) for source in relation.sources
        ]
        if relation.op == "multiply":
            result = probs[0]
            for value in probs[1:]:
                result = result * value
            return result
        if relation.op == "add":
            result = probs[0]
            for value in probs[1:]:
                result = result + value
            return result
        if relation.op == "subtract":
            if len(probs) != 2:
                raise ValueError(f"Relation '{relation.target}' subtract requires 2 sources")
            return probs[0] - probs[1]
        if relation.op == "divide":
            if len(probs) != 2:
                raise ValueError(f"Relation '{relation.target}' divide requires 2 sources")
            return probs[0] / (probs[1] + 1e-8)
        raise ValueError(f"Unknown relation op: {relation.op}")
