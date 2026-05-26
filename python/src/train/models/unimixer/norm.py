from __future__ import annotations

"""SiameseNorm：双流归一化 RMSNorm + 融合。"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        ) * self.weight


class SiameseNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.rmsnorm = RMSNorm(normalized_shape, eps)

    def forward_rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x)

    def forward(
        self,
        x_bar: torch.Tensor,
        y_bar: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if output is not None:
            return self.rmsnorm(x_bar + output), y_bar + output
        else:
            return x_bar + self.rmsnorm(y_bar)
