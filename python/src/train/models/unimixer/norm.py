from __future__ import annotations

"""SiameseNorm：双流归一化 RMSNorm + 融合。"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        return (
            x / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        ) * self.weight


class SiameseNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.rmsnorm = RMSNorm(normalized_shape, eps)

    def forward_rmsnorm(self, x):
        return self.rmsnorm(x)

    def forward(self, x_bar, y_bar, output=None):
        if output is not None:
            return self.rmsnorm(x_bar + output), y_bar + output
        else:
            return x_bar + self.rmsnorm(y_bar)
