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
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation
from .esmm import _probability_for_relation
from .output_head import _execute_relation


class GateNU(nn.Module):
    """Gate Neural Unit: ReLU hidden layer followed by gamma-scaled sigmoid gate."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, gamma: float = 2.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class PersonalizedTaskTower(nn.Module):
    """Task tower with PPNet gates applied to every hidden layer."""

    def __init__(self, config: object, input_dim: int, prior_dim: int) -> None:
        super().__init__()
        self.name = config.name
        self.output_kind = config.output_kind if hasattr(config, "output_kind") else config.kind
        activation = config.activation
        self._act = activation if isinstance(activation, Activation) else Activation.from_str(activation)
        self.hidden = nn.ModuleDict()
        self.pp_gates = nn.ModuleDict()
        in_dim = input_dim
        for i, h_dim in enumerate(config.hidden_dims):
            key = str(i)
            self.hidden[key] = nn.Linear(in_dim, h_dim)
            self.pp_gates[key] = GateNU(prior_dim, prior_dim, h_dim)
            in_dim = h_dim
        n = len(config.hidden_dims)
        self.output = nn.ModuleDict()
        self.output[str(n)] = nn.Linear(in_dim, getattr(config, "output_dim", 1))

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.hidden)):
            key = str(i)
            x = self._act.apply(self.hidden[key](x)) * self.pp_gates[key](prior)
        return self.output[str(len(self.hidden))](x)


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
        ep_prior_features: list[str] | None = None,
        pp_prior_features: list[str] | None = None,
    ) -> None:
        super().__init__()
        deep_hidden_dims = deep_hidden_dims or []
        shared_bottom_dims = shared_bottom_dims or []

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        total_dim = self.embeddings.total_dim
        self.ep_prior_indices = self._prior_indices(ep_prior_features)
        self.pp_prior_indices = self._prior_indices(pp_prior_features)

        # Explicit personalized priors selected by feature name.
        self.ep_prior_proj = nn.Linear(len(self.ep_prior_indices), prior_dim, bias=False)
        self.pp_prior_proj = nn.Linear(len(self.pp_prior_indices), prior_dim, bias=False)

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

        if output_contract is not None:
            self.output_contract = output_contract
            self.output_towers = nn.ModuleDict()
            self._tower_inputs: dict[str, str] = {}
            for tower in output_contract.towers:
                if tower.input != "shared":
                    raise ValueError("PEPNet output_contract towers must use input='shared'")
                self.output_towers[tower.name] = PersonalizedTaskTower(tower, shared_dim, prior_dim)
                self._tower_inputs[tower.name] = tower.input
            relations = {relation.name: relation for relation in output_contract.relations}
            self._relations = [relations[name] for name in output_contract.relation_order]
            self.task_names = [tower.name for tower in output_contract.towers]
            self.relation_names = list(output_contract.relation_order)
            return

        self.task_config = task_config
        if task_config is None:
            raise ValueError("PEPNet requires task_config or output_contract")
        self.task_names = [tower.name for tower in self.task_config.towers]
        self.relation_names = [relation.target for relation in self.task_config.relations]
        for tower in self.task_config.towers:
            setattr(self, f"{tower.name}_tower", PersonalizedTaskTower(tower, shared_dim, prior_dim))

    def _prior_indices(self, names: list[str] | None) -> list[int]:
        if not names:
            return list(range(self.embeddings.num_features))
        unknown = [name for name in names if name not in self.embeddings.feature_to_idx]
        if unknown:
            raise ValueError(f"PEPNet prior feature(s) not embeddable: {unknown}")
        return [self.embeddings.feature_to_idx[name] for name in names]

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_contract"):
            return self.forward_execution(x_inputs).outputs
        shared, pp_prior = self._shared(x_inputs)
        return self._forward_legacy(shared, pp_prior)

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        shared, pp_prior = self._shared(x_inputs)
        if hasattr(self, "output_contract"):
            return self._forward_contract(shared, pp_prior)
        outputs = self._forward_legacy(shared, pp_prior)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _shared(self, x_inputs: FeatureTensorMap) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = self.embeddings.forward_stacked(x_inputs)
        # stacked: list of [batch, 1, dim_i]

        ep_prior = self.ep_prior_proj(self._prior_raw(stacked, self.ep_prior_indices))
        pp_prior = self.pp_prior_proj(self._prior_raw(stacked, self.pp_prior_indices))

        # EPNet gate on embeddings
        epnet_scale = self.epnet_gate(ep_prior)  # [batch, total_dim]
        dense_concat = torch.cat([e.squeeze(1) for e in stacked], dim=1)  # [batch, total_dim]
        gated = dense_concat * epnet_scale

        shared = self.deep(gated) if self.has_deep else gated
        if hasattr(self, "shared_bottom"):
            shared = self.shared_bottom(shared)

        return shared, pp_prior

    @staticmethod
    def _prior_raw(stacked: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
        prior_parts = [stacked[index].mean(dim=2, keepdim=True) for index in indices]
        prior_raw = torch.cat(prior_parts, dim=1)
        return prior_raw.squeeze(2).detach()

    def _forward_contract(self, shared: torch.Tensor, pp_prior: torch.Tensor) -> ModelExecution:
        nodes = ModelOutput()
        for name, tower in self.output_towers.items():
            nodes.insert(name, tower(shared, pp_prior), tower.output_kind)
        for relation in self._relations:
            tensor, kind = _execute_relation(relation, nodes)
            nodes.insert(relation.name, tensor, kind)
        outputs = ModelOutput()
        for output in self.output_contract.outputs:
            source = nodes.get(output.source)
            if source is None:
                raise ValueError(f"public output '{output.name}' source '{output.source}' is missing")
            outputs.insert(output.name, source.tensor, source.kind)
        return ModelExecution(nodes=nodes, outputs=outputs)

    def _forward_legacy(self, shared: torch.Tensor, pp_prior: torch.Tensor) -> ModelOutput:
        outputs = ModelOutput()
        for name in self.task_names:
            tower = getattr(self, f"{name}_tower")
            outputs.insert(name, tower(shared, pp_prior), tower.output_kind)
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
