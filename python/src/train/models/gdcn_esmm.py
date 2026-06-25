from __future__ import annotations

"""GDCN + ESMM: gated cross network shared representation with ESMM task towers."""

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
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TaskTower
from .esmm import _probability_for_relation, default_task_config
from .output_head import OutputHead


class GDCNESMM(nn.Module):
    """ESMM variant using a gated cross network plus optional deep branch."""

    def __init__(
        self,
        features: list[FeatureTuple],
        cross_layers: int = 3,
        deep_hidden_dims: list[int] | None = None,
        shared_bottom_dims: list[int] | None = None,
        click_hidden_dims: list[int] | None = None,
        cvr_hidden_dims: list[int] | None = None,
        detail_hidden_dims: list[int] | None = None,
        stock_hidden_dims: list[int] | None = None,
        stay_hidden_dims: list[int] | None = None,
        task_config: MultiTaskConfig | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
        super().__init__()
        deep_hidden_dims = deep_hidden_dims or []
        shared_bottom_dims = shared_bottom_dims or []
        click_hidden_dims = click_hidden_dims or [8]
        cvr_hidden_dims = cvr_hidden_dims or [8]
        detail_hidden_dims = detail_hidden_dims or [8]
        stock_hidden_dims = stock_hidden_dims or [8]
        stay_hidden_dims = stay_hidden_dims or [8]

        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        input_dim = self.embeddings.total_dim
        self.cross = GatedCrossNetwork(input_dim, cross_layers)
        self.has_deep = bool(deep_hidden_dims)
        if self.has_deep:
            self.deep = Mlp(input_dim, deep_hidden_dims[:-1], deep_hidden_dims[-1], Activation.RELU)
            fusion_dim = input_dim + deep_hidden_dims[-1]
        else:
            fusion_dim = input_dim

        if shared_bottom_dims:
            self.shared_bottom = Mlp(
                fusion_dim,
                shared_bottom_dims[:-1],
                shared_bottom_dims[-1],
                Activation.RELU,
            )
            tower_input_dim = shared_bottom_dims[-1]
        else:
            tower_input_dim = fusion_dim

        if output_contract is not None:
            self.output_head = OutputHead(output_contract, {"shared": tower_input_dim})
            self.task_names = [tower.name for tower in output_contract.towers]
            self.relation_names = list(output_contract.relation_order)
            return

        self.task_config = task_config or default_task_config(
            click_hidden_dims,
            cvr_hidden_dims,
            detail_hidden_dims,
            stock_hidden_dims,
            stay_hidden_dims,
        )
        self.task_names = [tower.name for tower in self.task_config.towers]
        self.relation_names = [relation.target for relation in self.task_config.relations]
        for tower in self.task_config.towers:
            setattr(self, f"{tower.name}_tower", TaskTower(tower, tower_input_dim))

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
        dense = self.embeddings(x_inputs)
        cross_out = self.cross(dense)
        shared = torch.cat([cross_out, self.deep(dense)], dim=1) if self.has_deep else cross_out
        if hasattr(self, "shared_bottom"):
            shared = self.shared_bottom(shared)
        return shared

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
