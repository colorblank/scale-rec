from __future__ import annotations

"""损失函数与评估指标 — 纯函数，无副作用。"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .dag import FeatureDag

# 模型 Batch 字典类型
Batch = dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════


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


def _bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """标准 binary cross entropy with logits。"""
    return F.binary_cross_entropy_with_logits(logits, labels)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list[Any]],
    label_map: dict[str, str],
) -> torch.Tensor | None:
    """多任务损失：click/cvr/detail/stock→BCE，stay→weighted BCE。"""
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
            loss = _bce(logits[valid], y)

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
