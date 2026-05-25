from __future__ import annotations

"""多任务不确定性加权损失 (MultiTaskLoss)。"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_labels(batch_labels: dict[str, list[Any]], *names: str) -> list[Any]:
    for n in names:
        vals = batch_labels.get(n)
        if vals and any(v is not None for v in vals):
            return vals
    return []


def _to_device(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


def _weighted_bce_stay(logits: torch.Tensor, stay_times: torch.Tensor) -> torch.Tensor:
    t = stay_times.float()
    mask = (t >= 0).squeeze(-1)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    z, t = logits[mask], t[mask]
    return -(t / (1 + t) * (-F.softplus(-z)) + 1 / (1 + t) * (-F.softplus(z))).mean()


class MultiTaskLoss(nn.Module):
    """多任务损失：支持 static / equal / uncertainty 模式。"""

    def __init__(
        self,
        task_names: list[str],
        label_map: dict[str, str],
        *,
        mode: str = "static",
        task_weights: dict[str, float] | None = None,
        pos_weights: dict[str, float] | None = None,
        reg_weight: float = 0.1,
    ):
        super().__init__()
        if mode not in {"static", "equal", "uncertainty"}:
            raise ValueError(f"Unknown loss weighting mode: {mode}")
        self.task_names = [t for t in task_names if not t.startswith("ct")]
        self.label_map = label_map
        self.mode = mode
        self.task_weights = task_weights or {}
        self.pos_weights = pos_weights or {}
        self.reg_weight = reg_weight
        if mode == "uncertainty":
            self.log_vars = nn.ParameterDict(
                {t: nn.Parameter(torch.zeros(1)) for t in self.task_names}
            )
        else:
            self.log_vars = nn.ParameterDict()

    def forward(
        self, outputs: dict[str, torch.Tensor], batch_labels: dict[str, list[Any]]
    ) -> torch.Tensor | None:
        total: torch.Tensor | None = None
        self._last_raw_losses: dict[str, float] = {}
        self._last_pos_rates: dict[str, float] = {}
        for task, logits in outputs.items():
            if task.startswith("ct"):
                continue
            col = self.label_map.get(task)
            if not col:
                continue
            raw = _pick_labels(batch_labels, col, task)
            if not raw:
                continue
            arr = np.array([float(v) if v is not None else np.nan for v in raw], dtype=np.float32)
            valid = ~np.isnan(arr)
            if not valid.any():
                continue

            if task == "stay":
                t = torch.tensor(arr[valid], dtype=torch.float32, device=logits.device).view(-1, 1)
                raw_loss = _weighted_bce_stay(logits[valid], t)
            else:
                y = torch.tensor(arr[valid], dtype=torch.float32, device=logits.device).view(-1, 1)
                pw = self.pos_weights.get(task)
                if pw is not None:
                    raw_loss = F.binary_cross_entropy_with_logits(
                        logits[valid], y, pos_weight=torch.tensor(pw, device=logits.device)
                    )
                else:
                    raw_loss = F.binary_cross_entropy_with_logits(logits[valid], y)

            self._last_raw_losses[task] = float(raw_loss.detach().item())
            self._last_pos_rates[task] = float(arr[valid].mean())

            w = 1.0 if self.mode == "equal" else self.task_weights.get(task, 1.0)
            if self.mode == "uncertainty":
                lv = self.log_vars.get(task)
                if lv is not None:
                    lv_c = torch.clamp(lv, -3.0, 3.0)
                    loss = w * (torch.exp(-lv_c) * raw_loss + self.reg_weight * lv_c)
                else:
                    loss = w * raw_loss
            else:
                loss = w * raw_loss
            total = loss if total is None else total + loss
        return total

    def last_losses(self) -> dict[str, float]:
        return getattr(self, "_last_raw_losses", {})

    def last_pos_rates(self) -> dict[str, float]:
        return getattr(self, "_last_pos_rates", {})

    def task_weights_info(self) -> dict[str, float]:
        if self.mode == "uncertainty":
            return {
                t: float(torch.exp(-torch.clamp(self.log_vars[t], -3.0, 3.0)).item())
                for t in self.log_vars
            }
        if self.mode == "equal":
            return {t: 1.0 for t in self.task_names}
        return {t: self.task_weights.get(t, 1.0) for t in self.task_names}


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list[Any]],
    label_map: dict[str, str],
) -> torch.Tensor | None:
    return MultiTaskLoss(list(outputs), label_map, mode="equal")(outputs, batch_labels)
