from __future__ import annotations

"""GDCN + ESMM: gated cross network shared representation with ESMM task towers."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.gdcn import GatedCrossNetwork
from ..layers.mlp import Mlp
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TaskTower
from .esmm import default_task_config


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

    def forward(self, x_inputs: FeatureTensorMap) -> dict[str, torch.Tensor]:
        dense = self.embeddings(x_inputs)
        cross_out = self.cross(dense)
        shared = torch.cat([cross_out, self.deep(dense)], dim=1) if self.has_deep else cross_out
        if hasattr(self, "shared_bottom"):
            shared = self.shared_bottom(shared)

        outputs = {name: getattr(self, f"{name}_tower")(shared) for name in self.task_names}
        for relation in self.task_config.relations:
            outputs[relation.target] = self._apply_relation(relation, outputs)
        return outputs

    @staticmethod
    def _apply_relation(relation: TaskRelation, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not relation.sources:
            raise ValueError(f"Relation '{relation.target}' has no sources")
        probs = [torch.sigmoid(outputs[source]) for source in relation.sources]
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
