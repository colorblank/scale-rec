import torch.nn as nn
from .towers import Activation


class Mlp(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation):
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

    def forward(self, x):
        for i in range(len(self.hidden)):
            x = self.activation.apply(self.hidden[str(i)](x))
        return self.output(x)
