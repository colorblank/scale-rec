from __future__ import annotations

"""Configurable ESMM: shared bottom + task towers + probability relations."""

import torch
import torch.nn as nn

from ..layers.embedding import FeatureEmbeddings
from ..layers.mlp import Mlp
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TaskTower, TowerConfig


def default_task_config(
    click_hidden_dims,
    cvr_hidden_dims,
    detail_hidden_dims,
    stock_hidden_dims,
    stay_hidden_dims,
) -> MultiTaskConfig:
    return MultiTaskConfig(
        towers=[
            TowerConfig("click", click_hidden_dims, 1, Activation.RELU),
            TowerConfig("cvr", cvr_hidden_dims, 1, Activation.RELU),
            TowerConfig("detail", detail_hidden_dims, 1, Activation.RELU),
            TowerConfig("stock", stock_hidden_dims, 1, Activation.RELU),
            TowerConfig("stay", stay_hidden_dims, 1, Activation.RELU),
        ],
        relations=[
            TaskRelation("ctcvr", ["click", "cvr"], "multiply"),
            TaskRelation("ctdetail", ["click", "detail"], "multiply"),
            TaskRelation("ctstock", ["click", "stock"], "multiply"),
            TaskRelation("ctstay", ["detail", "stay"], "multiply"),
        ],
    )


class ESMM(nn.Module):
    """Entire-space multi-task model with configurable towers and relations."""

    def __init__(
        self,
        features,
        shared_bottom_dims,
        click_hidden_dims,
        cvr_hidden_dims,
        detail_hidden_dims,
        stock_hidden_dims,
        stay_hidden_dims,
        task_config: MultiTaskConfig | None = None,
        pooling_map=None,
        total_dim=None,
    ):
        super().__init__()
        self.embeddings = FeatureEmbeddings(features, pooling_map, total_dim=total_dim)
        if shared_bottom_dims:
            self.shared_bottom = Mlp(
                self.embeddings.total_dim,
                shared_bottom_dims[:-1],
                shared_bottom_dims[-1],
                Activation.RELU,
            )
            sd = shared_bottom_dims[-1]
        else:
            sd = self.embeddings.total_dim

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
            setattr(self, f"{tower.name}_tower", TaskTower(tower, sd))

    def forward(self, x_inputs):
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat

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
