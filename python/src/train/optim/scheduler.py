from __future__ import annotations

"""Step-based warmup → cosine decay LR scheduler."""
import math

import torch


class LRScheduler:
    def __init__(
        self,
        optimizers: list[torch.optim.Optimizer],
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float,
    ):
        self.optimizers = optimizers
        self.warmup = warmup_steps
        self.total = max(total_steps, warmup_steps + 1)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [
            [float(pg["lr"]) for pg in optimizer.param_groups] for optimizer in optimizers
        ]

    def step(self, step: int) -> None:
        if step <= self.warmup:
            scale = step / max(self.warmup, 1)
        else:
            progress = min((step - self.warmup) / max(self.total - self.warmup, 1), 1.0)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )
        for optimizer, base_lrs in zip(self.optimizers, self.base_lrs, strict=True):
            for pg, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
                pg["lr"] = base_lr * scale

    def current_lr(self) -> float:
        return self.optimizers[0].param_groups[0]["lr"]


def build_optimizer(params_groups: list[dict], name: str) -> torch.optim.Optimizer:
    """从名称构建优化器。"""
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW([g for g in params_groups if g["params"]])
    elif name == "adam":
        return torch.optim.Adam([g for g in params_groups if g["params"]])
    elif name == "sgd":
        momentum = params_groups[0].get("momentum", 0)
        return torch.optim.SGD([g for g in params_groups if g["params"]], momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
