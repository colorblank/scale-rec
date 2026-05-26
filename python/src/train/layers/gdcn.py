from __future__ import annotations

"""Gated Deep & Cross Network layers."""

import torch
import torch.nn as nn


class GatedCrossNetwork(nn.Module):
    """Gated cross network over dense feature vectors.

    Args:
        input_dim: Dense input dimension.
        num_layers: Number of gated cross layers.
    """

    def __init__(self, input_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cross = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for bias in self.bias:
            nn.init.uniform_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated cross layers to a [batch, input_dim] tensor."""
        x0 = x
        xi = x
        for i in range(self.num_layers):
            weight = torch.sigmoid(self.gate[i](xi))
            crossed = self.cross[i](xi) + self.bias[i]
            xi = x0 * crossed * weight + xi
        return xi
