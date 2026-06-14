from __future__ import annotations

"""MMoE：多门控专家混合。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embedding import FeatureEmbeddings, FeatureTensorMap, FeatureTuple
from ..layers.mlp import Mlp
from ..layers.towers import Activation


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts for multi-task learning."""

    def __init__(
        self,
        features: list[FeatureTuple],
        shared_bottom_dims: list[int],
        num_experts: int,
        expert_hidden_dims: list[int],
        expert_output_dim: int,
        task_configs: list[tuple[str, list[int]]],
        pooling_map: dict[str, str] | None = None,
        total_dim: int | None = None,
    ) -> None:
        """Build MMoE: embeddings + optional shared_bottom + N experts + K gates + K towers."""
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
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim

        self._experts = []
        for e in range(num_experts):
            expert = Mlp(sd, expert_hidden_dims, expert_output_dim, Activation.RELU)
            setattr(self, f"expert_{e}", expert)
            self._experts.append(expert)

        self._gates = []
        for t in range(len(task_configs)):
            gate = nn.Linear(sd, num_experts)
            setattr(self, f"gate_{t}", gate)
            self._gates.append(gate)

        self.task_names = []
        self._towers = []
        for t, (name, tower_dims) in enumerate(task_configs):
            self.task_names.append(name)
            tower = Mlp(expert_output_dim, tower_dims, 1, Activation.RELU)
            setattr(self, f"task_{t}_tower", tower)
            self._towers.append(tower)

    def forward(self, x_inputs: FeatureTensorMap) -> dict[str, torch.Tensor]:
        """Forward: embed -> shared -> experts -> gate softmax -> weighted sum -> task towers."""
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat
        experts = torch.cat([e(shared).unsqueeze(1) for e in self._experts], dim=1)
        outputs = {}
        for t in range(len(self.task_names)):
            g = F.softmax(self._gates[t](shared), dim=1)
            gated = (experts * g.unsqueeze(2)).sum(dim=1)
            outputs[self.task_names[t]] = self._towers[t](gated)
        return outputs
