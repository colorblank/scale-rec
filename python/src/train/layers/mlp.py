from __future__ import annotations

"""通用多层感知机模块 — mirrors src/layers/mlp.rs."""
import torch
import torch.nn as nn

from .towers import Activation


class Mlp(nn.Module):
    """Generic MLP with inter-layer activation, no final activation.

    Parameter naming uses `nn.ModuleDict` so that state_dict keys match
    Candle's `vb.pp("hidden.0")` and `vb.pp("output")` paths exactly.
    When `hidden_dims` is empty, degenerates to a single `nn.Linear`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: Activation,
    ):
        """Construct MLP.

        Args:
            input_dim: Input feature dimension.
            hidden_dims: Hidden layer dimensions (empty for single linear).
            output_dim: Output dimension (logits).
            activation: Inter-layer activation; final layer has no activation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.hidden = nn.ModuleDict()
        in_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.hidden[str(i)] = nn.Linear(in_dim, h_dim)
            in_dim = h_dim
        self.output = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: hidden layers with activation, then output layer without."""
        for i in range(len(self.hidden)):
            x = self.activation.apply(self.hidden[str(i)](x))
        return self.output(x)
