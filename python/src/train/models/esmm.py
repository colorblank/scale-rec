from __future__ import annotations

"""Configurable ESMM: shared bottom + task towers + probability relations."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.config import PoolingMode

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract
from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TaskTower, TowerConfig
from .output_head import OutputHead


def default_task_config(
    click_hidden_dims: list[int],
    cvr_hidden_dims: list[int],
    detail_hidden_dims: list[int],
    stock_hidden_dims: list[int],
    stay_hidden_dims: list[int],
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
        features: list[FeatureTuple],
        shared_bottom_dims: list[int],
        click_hidden_dims: list[int],
        cvr_hidden_dims: list[int],
        detail_hidden_dims: list[int],
        stock_hidden_dims: list[int],
        stay_hidden_dims: list[int],
        task_config: MultiTaskConfig | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
        output_contract: NormalizedOutputContract | None = None,
    ) -> None:
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

        if output_contract is not None:
            self.output_contract = output_contract
            self.output_head = OutputHead(output_contract, {"shared": sd})
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
            setattr(self, f"{tower.name}_tower", TaskTower(tower, sd))

    def forward(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        if hasattr(self, "output_head"):
            return self.forward_execution(x_inputs).outputs
        return self._forward_legacy(x_inputs)

    def forward_execution(self, x_inputs: FeatureTensorMap) -> ModelExecution:
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat
        if hasattr(self, "output_head"):
            return self.output_head({"shared": shared})
        outputs = self._forward_legacy_from_shared(shared)
        return ModelExecution(nodes=outputs, outputs=outputs)

    def _forward_legacy(self, x_inputs: FeatureTensorMap) -> ModelOutput:
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat
        return self._forward_legacy_from_shared(shared)

    def _forward_legacy_from_shared(self, shared: torch.Tensor) -> ModelOutput:
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


def _probability_for_relation(
    relation: TaskRelation, outputs: ModelOutput, source: str
) -> torch.Tensor:
    output = outputs.get(source)
    if output is None:
        raise ValueError(f"Relation '{relation.target}' source '{source}' is missing")
    if output.kind == "binary_logit":
        return torch.sigmoid(output.tensor)
    if output.kind == "probability":
        return output.tensor
    raise ValueError(
        f"Relation '{relation.target}' source '{source}' must be binary_logit or probability, "
        f"got {output.kind}"
    )
