from __future__ import annotations

"""FM 二阶交互：0.5 * Σ[(Σv_i)² - Σv_i²]。"""

import torch


def fm_interaction(stacked: torch.Tensor) -> torch.Tensor:
    sum_square = stacked.pow(2).sum(dim=1)
    square_sum = stacked.sum(dim=1).pow(2)
    return 0.5 * (square_sum - sum_square).sum(dim=1, keepdim=True)
