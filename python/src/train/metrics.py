from __future__ import annotations

"""损失函数与评估指标。"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dag import FeatureDag

Batch = dict[str, Any]


def _pick_labels(batch_labels: dict[str, list[Any]], *names: str) -> list[Any]:
    """按优先级从 batch 标签中取值，跳过全为 None 的列。"""
    for n in names:
        vals = batch_labels.get(n)
        if vals and any(v is not None for v in vals):
            return vals
    return []


def _weighted_bce_stay(logits: torch.Tensor, stay_times: torch.Tensor) -> torch.Tensor:
    """stay_time 加权 BCE: -(t/(1+t))·log σ(z) - (1/(1+t))·log(1-σ(z))。"""
    t = stay_times.float()
    mask = (t >= 0).squeeze(-1)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    z, t = logits[mask], t[mask]
    log_p = -F.softplus(-z)
    log_1mp = -F.softplus(z)
    return -(t / (1 + t) * log_p + 1 / (1 + t) * log_1mp).mean()


class MultiTaskLoss(nn.Module):
    """多任务损失：支持 uncertainty weighting / static / equal 三种模式。

    - "uncertainty": Kendall 2018, L = Σ w * [exp(-log_var) * L + λ * log_var]
      log_var clamped 到 [-3, 3] 防止权重极端化
    - "static": 固定任务权重 L = Σ w_i * L_i
    - "equal": 等权加和 L = Σ L_i
    """

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
        self.task_names = [t for t in task_names if t != "stay"]
        self.label_map = label_map
        self.mode = mode
        self.task_weights = task_weights or {}
        self.pos_weights = pos_weights or {}
        self.reg_weight = reg_weight

        if mode == "uncertainty":
            self.log_vars = nn.ParameterDict()
            for t in self.task_names:
                self.log_vars[t] = nn.Parameter(torch.zeros(1))
            if "stay" in task_names:
                self.log_vars["stay"] = nn.Parameter(torch.zeros(1))
        else:
            self.log_vars = nn.ParameterDict()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch_labels: dict[str, list[Any]],
    ) -> torch.Tensor | None:
        total: torch.Tensor | None = None
        self._last_raw_losses: dict[str, float] = {}
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
                        logits[valid], y,
                        pos_weight=torch.tensor(pw, device=logits.device),
                    )
                else:
                    raw_loss = F.binary_cross_entropy_with_logits(logits[valid], y)

            self._last_raw_losses[task] = float(raw_loss.detach().item())

            weight = self.task_weights.get(task, 1.0)
            if self.mode == "uncertainty":
                log_var = self.log_vars.get(task)
                if log_var is not None:
                    log_var_clamped = torch.clamp(log_var, -3.0, 3.0)
                    precision = torch.exp(-log_var_clamped)
                    loss = weight * (precision * raw_loss + self.reg_weight * log_var_clamped)
                else:
                    loss = weight * raw_loss
            else:
                loss = weight * raw_loss

            total = loss if total is None else total + loss
        return total

    def last_losses(self) -> dict[str, float]:
        """返回最近一次 forward 各任务的原始 loss 值，用于 debug。"""
        return getattr(self, "_last_raw_losses", {})

    def task_weights_info(self) -> dict[str, float]:
        """返回各任务当前有效权重。"""
        if self.mode == "uncertainty":
            return {
                t: float(torch.exp(-torch.clamp(self.log_vars[t], -3.0, 3.0)).item())
                for t in self.log_vars
            }
        return {t: self.task_weights.get(t, 1.0) for t in self.task_names}


# ═══════════════════════════════════════════════════════════════════
# Legacy: 简单等权加和，供需要确定性行为的场景使用
# ═══════════════════════════════════════════════════════════════════


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list[Any]],
    label_map: dict[str, str],
) -> torch.Tensor | None:
    """简单等权多任务损失（无 uncertainty weighting）。"""
    total: torch.Tensor | None = None
    for task, logits in outputs.items():
        if task.startswith("ct"):
            continue
        col = label_map.get(task)
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
            loss = _weighted_bce_stay(logits[valid], t)
        else:
            y = torch.tensor(arr[valid], dtype=torch.float32, device=logits.device).view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(logits[valid], y)

        total = loss if total is None else total + loss
    return total


# ═══════════════════════════════════════════════════════════════════
# AUC
# ═══════════════════════════════════════════════════════════════════


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """AUC via sorting: P(positive ranks above negative)."""
    order = np.argsort(probs)[::-1]
    y = labels[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_neg = np.cumsum(1 - y)
    pos_mask = y == 1
    return float(np.sum(n_neg - cum_neg[pos_mask]) / (n_pos * n_neg))


def _to_device(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


def compute_aucs(
    model: torch.nn.Module,
    dag: FeatureDag,
    batches: list[Batch],
    task_names: list[str],
    label_map: dict[str, str],
    device: torch.device | None = None,
) -> dict[str, float]:
    """在多个 batch 上计算各分类任务的 AUC。"""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    logits_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
    labels_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
    with torch.no_grad():
        for batch in batches:
            rows = [{k: v for k, v in r.items() if v is not None} for r in batch["features"]]
            outputs = model(_to_device(dag.preprocess_batch(rows), device))
            for t in task_names:
                if t not in outputs:
                    continue
                logits_buf[t].append(outputs[t].cpu().numpy().flatten())
                col = label_map.get(t, t)
                raw = _pick_labels(batch["labels"], col, t)
                arr = np.array(
                    [float(v) if v is not None else np.nan for v in raw], dtype=np.float32
                )
                labels_buf[t].append(arr[~np.isnan(arr)])
    if was_training:
        model.train()
    aucs: dict[str, float] = {}
    for t in task_names:
        if logits_buf[t] and labels_buf[t]:
            y = np.concatenate(labels_buf[t])
            if len(y):
                aucs[t] = _auc(y, _sigmoid(np.concatenate(logits_buf[t])))
    return aucs
