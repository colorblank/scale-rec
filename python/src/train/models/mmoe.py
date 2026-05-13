from __future__ import annotations

"""MMoE：多门控专家混合。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embedding import FeatureEmbeddings
from ..layers.mlp import Mlp
from ..layers.towers import Activation


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts for multi-task learning."""

    def __init__(
        self,
        features,
        shared_bottom_dims,
        num_experts,
        expert_hidden_dims,
        expert_output_dim,
        task_configs,
    ):
        super().__init__()
        self.embeddings = FeatureEmbeddings(features)
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
        for e in range(num_experts):
            setattr(
                self,
                f"expert_{e}",
                Mlp(sd, expert_hidden_dims, expert_output_dim, Activation.RELU),
            )
        self.num_tasks = len(task_configs)
        self.task_names = []
        for t, (name, tower_dims) in enumerate(task_configs):
            self.task_names.append(name)
            setattr(self, f"gate_{t}", nn.Linear(sd, num_experts))
            setattr(
                self,
                f"task_{t}_tower",
                Mlp(expert_output_dim, tower_dims, 1, Activation.RELU),
            )

    def forward(self, x_inputs):
        concat = self.embeddings(x_inputs)
        shared = self.shared_bottom(concat) if hasattr(self, "shared_bottom") else concat
        experts = torch.cat(
            [getattr(self, f"expert_{e}")(shared).unsqueeze(1) for e in range(self.num_experts)],
            dim=1,
        )
        outputs = {}
        for t in range(self.num_tasks):
            g = F.softmax(getattr(self, f"gate_{t}")(shared), dim=1)
            gated = (experts * g.unsqueeze(2)).sum(dim=1)
            outputs[self.task_names[t]] = getattr(self, f"task_{t}_tower")(gated)
        return outputs
