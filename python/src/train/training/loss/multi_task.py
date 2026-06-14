from __future__ import annotations

"""多任务不确定性加权损失 (MultiTaskLoss)。"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.task import TaskSpec, legacy_task_specs


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
        task_specs: list[TaskSpec] | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"static", "equal", "uncertainty"}:
            raise ValueError(f"Unknown loss weighting mode: {mode}")
        self.task_specs = task_specs or legacy_task_specs(task_names, label_map, task_weights)
        self.spec_by_name = {spec.name: spec for spec in self.task_specs}
        self.task_names = [spec.name for spec in self.task_specs]
        self.label_map = label_map
        self.mode = mode
        self.task_weights = {spec.name: spec.weight for spec in self.task_specs}
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
        unknown_outputs = sorted(
            task for task in outputs if task not in self.spec_by_name and not task.startswith("ct")
        )
        if unknown_outputs:
            raise ValueError(
                "Model produced outputs not covered by task specs: " + ", ".join(unknown_outputs)
            )
        missing_outputs = sorted(task for task in self.task_names if task not in outputs)
        if missing_outputs:
            raise ValueError(
                "Task specs require model outputs that are missing: " + ", ".join(missing_outputs)
            )
        for spec in self.task_specs:
            task = spec.name
            logits = outputs[task]
            col = spec.label
            if col not in batch_labels and task not in batch_labels:
                raise ValueError(f"Task '{task}' label column '{col}' is missing from batch labels")
            raw = _pick_labels(batch_labels, col, task)
            if not raw:
                continue
            arr = np.array([float(v) if v is not None else np.nan for v in raw], dtype=np.float32)
            valid = ~np.isnan(arr)
            if spec.mask:
                valid &= _eval_mask(spec.mask, batch_labels, col)
            if not valid.any():
                continue

            if spec.loss == "weighted_bce_stay":
                t = torch.tensor(arr[valid], dtype=torch.float32, device=logits.device).view(-1, 1)
                raw_loss = _weighted_bce_stay(logits[valid], t)
            else:
                y = torch.tensor(arr[valid], dtype=torch.float32, device=logits.device).view(-1, 1)
                pw = spec.pos_weight if spec.pos_weight is not None else self.pos_weights.get(task)
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
            return dict.fromkeys(self.task_names, 1.0)
        return {t: self.task_weights.get(t, 1.0) for t in self.task_names}


def _eval_mask(mask: str, batch_labels: dict[str, list[Any]], default_col: str) -> np.ndarray:
    tokens = mask.strip().split()
    if len(tokens) != 3:
        raise ValueError(f"Unsupported task mask: {mask}")
    col, op, raw_threshold = tokens
    values = batch_labels.get(col) or batch_labels.get(default_col) or []
    arr = np.array([float(v) if v is not None else np.nan for v in values], dtype=np.float32)
    threshold = float(raw_threshold)
    if op == ">=":
        return arr >= threshold
    if op == ">":
        return arr > threshold
    if op == "<=":
        return arr <= threshold
    if op == "<":
        return arr < threshold
    if op == "==":
        return arr == threshold
    if op == "!=":
        return arr != threshold
    raise ValueError(f"Unsupported task mask operator: {op}")


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list[Any]],
    label_map: dict[str, str],
) -> torch.Tensor | None:
    return MultiTaskLoss(list(outputs), label_map, mode="equal")(outputs, batch_labels)
