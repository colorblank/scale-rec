"""PEPNet: Parameter and Embedding Personalized Network.

Architecture:
  FeatureEmbeddings → EPNet (per-domain gates) → Deep MLP → shared_bottom → PPNet gate → towers

EPNet applies a separate element-wise gate per domain (paper §3.2), each conditioned
on the domain's own pooled prior features. PPNet applies per-layer gates inside each
task tower using a global prior.

Reference: PEPNet (Kuaishou 2023, KDD) — sec 3.2 Domain-Specific Embedding Personalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class _DomainInfo:
    name: str
    feature_indices: list[int] = field(default_factory=list)
    prior_indices: list[int] = field(default_factory=list)
    dim: int = 0


class PEPNet(nn.Module):
    """PEPNet: personalized embedding and parameter gating for multi-task learning.

    Supports two EPNet modes:
      - **Per-domain** (paper): each domain gets its own GateNU (default when ``domains`` is set).
      - **Single-gate** (fallback): one GateNU for the full embedding vector (backward compat).
    """

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
        domains: list[dict] | None = None,
    ) -> None:
        super().__init__()
        deep_hidden_dims = deep_hidden_dims or []
        shared_bottom_dims = shared_bottom_dims or []

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        total_dim = self.embeddings.total_dim
        self.pp_prior_indices = self._prior_indices(pp_prior_features)

        self._feat_dim: dict[str, int] = {name: dim for name, _, dim in features}

        if domains is not None:
            self.domains = self._build_domains(domains)
            self.ep_prior_projs = nn.ModuleDict()
            self.epnet_gates = nn.ModuleDict()
            for info in self.domains:
                self.ep_prior_projs[info.name] = nn.Linear(len(info.prior_indices), prior_dim, bias=False)
                self.epnet_gates[info.name] = GateNU(prior_dim, prior_dim, info.dim)
            self.ep_prior_proj = None
            self.epnet_gate = None
        else:
            self.domains = None
            self.ep_prior_indices = self._prior_indices(ep_prior_features)
            self.ep_prior_proj = nn.Linear(len(self.ep_prior_indices), prior_dim, bias=False)
            self.epnet_gate = GateNU(prior_dim, prior_dim, total_dim)

        self.pp_prior_proj = nn.Linear(len(self.pp_prior_indices), prior_dim, bias=False)

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

    def _build_domains(self, domains: list[dict]) -> list[_DomainInfo]:
        all_indices: set[int] = set()
        infos: list[_DomainInfo] = []
        for domain in domains:
            name = domain["name"]
            feat_names = domain["features"]
            unknown = [n for n in feat_names if n not in self.embeddings.feature_to_idx]
            if unknown:
                raise ValueError(f"PEPNet domain '{name}' unknown features: {unknown}")
            feature_indices = [self.embeddings.feature_to_idx[n] for n in feat_names]
            prior_names = domain.get("ep_prior_features", feat_names)
            unknown_prior = [n for n in prior_names if n not in self.embeddings.feature_to_idx]
            if unknown_prior:
                raise ValueError(f"PEPNet domain '{name}' unknown ep_prior_features: {unknown_prior}")
            prior_indices = [self.embeddings.feature_to_idx[n] for n in prior_names]
            dim = sum(self._feat_dim[n] for n in feat_names)
            infos.append(_DomainInfo(name, feature_indices, prior_indices, dim))
            all_indices.update(feature_indices)
        unassigned = set(range(self.embeddings.num_features)) - all_indices
        if unassigned:
            names = [self.embeddings.ordered_names[i] for i in unassigned]
            raise ValueError(f"PEPNet features not assigned to any domain: {names}")
        return infos

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

        pp_prior = self.pp_prior_proj(self._prior_raw(stacked, self.pp_prior_indices))

        if self.domains is not None:
            gated_parts = []
            for info in self.domains:
                domain_embs = [stacked[i] for i in info.feature_indices]
                domain_concat = torch.cat([e.squeeze(1) for e in domain_embs], dim=1)
                domain_prior = self.ep_prior_projs[info.name](
                    self._prior_raw(stacked, info.prior_indices)
                )
                domain_scale = self.epnet_gates[info.name](domain_prior)
                gated_parts.append(domain_concat * domain_scale)
            gated = torch.cat(gated_parts, dim=1)
        else:
            ep_prior = self.ep_prior_proj(self._prior_raw(stacked, self.ep_prior_indices))
            epnet_scale = self.epnet_gate(ep_prior)
            dense_concat = torch.cat([e.squeeze(1) for e in stacked], dim=1)
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
