from __future__ import annotations

"""多任务塔：TaskTower、MultiTaskTower、任务关系推导。"""
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.core.model_output import ModelOutput


class Activation(Enum):
    RELU = auto()
    SIGMOID = auto()
    SWISH = auto()
    GELU = auto()
    NONE = auto()

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function to tensor."""
        m = {
            self.RELU: F.relu,
            self.SIGMOID: torch.sigmoid,
            self.SWISH: lambda x: x * torch.sigmoid(x),
            self.GELU: F.gelu,
            self.NONE: lambda x: x,
        }
        return m[self](x)

    @classmethod
    def from_str(cls, s: str) -> Activation:
        """Parse activation from string: relu, sigmoid, swish, gelu, none."""
        return {
            "relu": cls.RELU,
            "sigmoid": cls.SIGMOID,
            "swish": cls.SWISH,
            "gelu": cls.GELU,
            "none": cls.NONE,
        }[s.lower()]


@dataclass
class TowerConfig:
    name: str
    hidden_dims: list[int] = field(default_factory=list)
    output_dim: int = 1
    activation: Activation = Activation.RELU
    output_kind: str = "binary_logit"


class TaskTower(nn.Module):
    """Single-task MLP tower. Naming matches Candle vb.pp paths."""

    def __init__(self, config: TowerConfig, input_dim: int) -> None:
        """Build tower from TowerConfig with hidden.{i} and output.{n} naming."""
        super().__init__()
        self.name = config.name
        self.output_kind = config.output_kind
        self._act = config.activation
        self.hidden = nn.ModuleDict()
        in_dim = input_dim
        for i, h in enumerate(config.hidden_dims):
            self.hidden[str(i)] = nn.Linear(in_dim, h)
            in_dim = h
        n = len(config.hidden_dims)
        self.output = nn.ModuleDict()
        self.output[str(n)] = nn.Linear(in_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: hidden layers with activation, output layer without."""
        for i in range(len(self.hidden)):
            x = self._act.apply(self.hidden[str(i)](x))
        return self.output[str(len(self.hidden))](x)


@dataclass
class TaskRelation:
    target: str
    sources: list[str]
    op: str


@dataclass
class MultiTaskConfig:
    towers: list[TowerConfig]
    relations: list[TaskRelation] = field(default_factory=list)


class MultiTaskTower(nn.Module):
    """Multi-task tower manager with optional task relation derivation."""

    def __init__(self, config: MultiTaskConfig, input_dim: int) -> None:
        """Build towers from config; each registered by name matching Candle vb.pp paths."""
        super().__init__()
        for tc in config.towers:
            setattr(self, tc.name, TaskTower(tc, input_dim))
        self._tower_names = [tc.name for tc in config.towers]
        self._relations = config.relations
        self._relation_names = [rel.target for rel in config.relations]

    def forward(self, shared: torch.Tensor) -> ModelOutput:
        """Run all towers, then apply task relations (sigmoid before op)."""
        outputs = ModelOutput()
        for name in self._tower_names:
            tower = getattr(self, name)
            outputs.insert(name, tower(shared), tower.output_kind)
        for rel in self._relations:
            probs = {s: _relation_probability(rel, outputs, s) for s in rel.sources}
            left = probs[rel.sources[0]]
            right = probs[rel.sources[1]]
            if rel.op == "multiply":
                outputs.insert_probability(rel.target, left * right)
            elif rel.op == "add":
                outputs.insert_probability(rel.target, left + right)
            elif rel.op == "subtract":
                outputs.insert_probability(rel.target, left - right)
            elif rel.op == "divide":
                outputs.insert_probability(rel.target, left / (right + 1e-8))
            else:
                raise ValueError(f"unsupported task relation op: {rel.op}")
        return outputs

    @property
    def relation_names(self) -> list[str]:
        return list(self._relation_names)


def _relation_probability(rel: TaskRelation, outputs: ModelOutput, source: str) -> torch.Tensor:
    output = outputs.get(source)
    if output is None:
        raise ValueError(f"Relation '{rel.target}' source '{source}' is missing")
    if output.kind == "binary_logit":
        return torch.sigmoid(output.tensor)
    if output.kind == "probability":
        return output.tensor
    raise ValueError(
        f"Relation '{rel.target}' source '{source}' must be binary_logit or probability, "
        f"got {output.kind}"
    )
