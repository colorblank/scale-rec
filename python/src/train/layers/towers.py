from __future__ import annotations

"""多任务塔：TaskTower、MultiTaskTower、任务关系推导。"""
from dataclasses import dataclass, field
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


class Activation(Enum):
    RELU = auto()
    SIGMOID = auto()
    SWISH = auto()
    GELU = auto()
    NONE = auto()

    def apply(self, x):
        m = {
            self.RELU: F.relu,
            self.SIGMOID: torch.sigmoid,
            self.SWISH: lambda x: x * torch.sigmoid(x),
            self.GELU: F.gelu,
            self.NONE: lambda x: x,
        }
        return m[self](x)

    @classmethod
    def from_str(cls, s):
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


class TaskTower(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.name = config.name
        self._act = config.activation
        self.hidden = nn.ModuleDict()
        in_dim = input_dim
        for i, h in enumerate(config.hidden_dims):
            self.hidden[str(i)] = nn.Linear(in_dim, h)
            in_dim = h
        n = len(config.hidden_dims)
        self.output = nn.ModuleDict()
        self.output[str(n)] = nn.Linear(in_dim, config.output_dim)

    def forward(self, x):
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
    def __init__(self, config, input_dim):
        super().__init__()
        for tc in config.towers:
            setattr(self, tc.name, TaskTower(tc, input_dim))
        self._tower_names = [tc.name for tc in config.towers]
        self._relations = config.relations

    def forward(self, shared):
        outputs = {n: getattr(self, n)(shared) for n in self._tower_names}
        for rel in self._relations:
            probs = {s: torch.sigmoid(outputs[s]) for s in rel.sources}
            m = {
                "multiply": lambda: probs[rel.sources[0]] * probs[rel.sources[1]],
                "add": lambda: probs[rel.sources[0]] + probs[rel.sources[1]],
                "subtract": lambda: probs[rel.sources[0]] - probs[rel.sources[1]],
                "divide": lambda: probs[rel.sources[0]] / (probs[rel.sources[1]] + 1e-8),
            }
            outputs[rel.target] = m[rel.op]()
        return outputs
