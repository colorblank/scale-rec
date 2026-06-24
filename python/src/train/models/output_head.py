from __future__ import annotations

"""Contract-driven task towers, relation graph execution and output projection."""

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..core.model_output import ModelExecution, ModelOutput
from ..core.output_contract import NormalizedOutputContract, RelationNode
from ..layers.towers import Activation, TaskTower, TowerConfig


class OutputHead(nn.Module):
    """Build and execute scalar task towers from an output contract."""

    def __init__(
        self,
        contract: NormalizedOutputContract,
        representation_dims: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.contract = contract
        self.towers = nn.ModuleDict()
        self._tower_inputs: dict[str, str] = {}
        for tower in contract.towers:
            if tower.input not in representation_dims:
                raise ValueError(
                    f"tower '{tower.name}' references unknown representation '{tower.input}'"
                )
            if representation_dims[tower.input] <= 0:
                raise ValueError(
                    f"representation '{tower.input}' dimension must be positive, "
                    f"got {representation_dims[tower.input]}"
                )
            self.towers[tower.name] = TaskTower(
                TowerConfig(
                    name=tower.name,
                    hidden_dims=list(tower.hidden_dims),
                    output_dim=1,
                    activation=Activation.from_str(tower.activation),
                    output_kind=tower.kind,
                ),
                representation_dims[tower.input],
            )
            self._tower_inputs[tower.name] = tower.input
        relations = {relation.name: relation for relation in contract.relations}
        self._relations = [relations[name] for name in contract.relation_order]

    def forward(self, representations: Mapping[str, torch.Tensor]) -> ModelExecution:
        nodes = ModelOutput()
        for name, tower in self.towers.items():
            input_name = self._tower_inputs[name]
            if input_name not in representations:
                raise ValueError(f"output head representation '{input_name}' is missing")
            nodes.insert(name, tower(representations[input_name]), tower.output_kind)
        for relation in self._relations:
            tensor, kind = _execute_relation(relation, nodes)
            nodes.insert(relation.name, tensor, kind)

        outputs = ModelOutput()
        for output in self.contract.outputs:
            source = nodes.get(output.source)
            if source is None:
                raise ValueError(
                    f"public output '{output.name}' source '{output.source}' is missing"
                )
            outputs.insert(output.name, source.tensor, source.kind)
        return ModelExecution(nodes=nodes, outputs=outputs)


def _execute_relation(relation: RelationNode, nodes: ModelOutput) -> tuple[torch.Tensor, str]:
    inputs = []
    for name in relation.inputs:
        value = nodes.get(name)
        if value is None:
            raise ValueError(f"relation '{relation.name}' input '{name}' is missing")
        inputs.append(value)

    if relation.op == "sigmoid":
        return torch.sigmoid(inputs[0].tensor), "probability"
    if relation.op == "multiply":
        result = inputs[0].tensor
        for value in inputs[1:]:
            result = result * value.tensor
        return result, "probability"
    if relation.op == "add":
        result = inputs[0].tensor
        for value in inputs[1:]:
            result = result + value.tensor
        return result, "regression"
    return inputs[0].tensor, inputs[0].kind
