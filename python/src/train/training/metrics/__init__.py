from __future__ import annotations

"""评估指标注册表与批量计算。"""
from typing import Any

import numpy as np
import torch

from ...core.model_output import ensure_model_output
from ...core.preprocessor import TrainingPreprocessor
from ..loss.multi_task import MultiTaskLoss as MultiTaskLoss
from ..loss.multi_task import _pick_labels, _to_device
from ..loss.multi_task import compute_loss as compute_loss
from .auc import _auc, _sigmoid, gauc
from .classification import accuracy, f1_score, logloss, recall
from .regression import mae, mse

Batch = dict[str, Any]

_ALL_METRICS = {
    "auc": ("AUC", lambda y, p: _auc(y, _sigmoid(p))),
    "gauc": ("GAUC", None),  # 需要 group_ids，特殊处理
    "logloss": ("LogLoss", lambda y, p: logloss(y, _sigmoid(p))),
    "acc": ("Accuracy", lambda y, p: accuracy(y, _sigmoid(p))),
    "recall": ("Recall", lambda y, p: recall(y, _sigmoid(p))),
    "f1": ("F1", lambda y, p: f1_score(y, _sigmoid(p))),
    "mae": ("MAE", mae),
    "mse": ("MSE", mse),
}


def get_available_metrics() -> list[str]:
    return list(_ALL_METRICS)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_names: list[str],
    group_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """批量计算多个指标。

    Args:
        y_true: 标签 (N,)。
        y_pred: 预测 logits (N,)。
        metric_names: 指标名列表。
        group_ids: GAUC 分组 ID，仅 gauc 需要。

    Returns:
        {metric_name: value}。
    """
    result: dict[str, float] = {}
    for m in metric_names:
        if m not in _ALL_METRICS:
            raise ValueError(f"Unknown metric: {m}")
        if m == "gauc":
            if group_ids is not None and len(group_ids) > 0:
                result[m] = gauc(y_true, _sigmoid(y_pred), group_ids)
            else:
                result[m] = 0.5
        else:
            fn = _ALL_METRICS[m][1]
            result[m] = float(fn(y_true, y_pred))
    return result


def compute_aucs(
    model: torch.nn.Module,
    preprocessor: TrainingPreprocessor,
    batches: list[Batch],
    task_names: list[str],
    label_map: dict[str, str],
    device: torch.device | None = None,
) -> dict[str, float]:
    """兼容旧 API：在多个 batch 上计算各分类任务的 AUC。"""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    logits_buf: dict[str, list[np.ndarray]] = {task: [] for task in task_names}
    labels_buf: dict[str, list[np.ndarray]] = {task: [] for task in task_names}
    with torch.no_grad():
        for batch in batches:
            rows = [{k: v for k, v in r.items() if v is not None} for r in batch["features"]]
            outputs = ensure_model_output(
                model(_to_device(preprocessor.preprocess_batch(rows), device))
            )
            for task in task_names:
                if task not in outputs:
                    continue
                col = label_map.get(task, task)
                raw = _pick_labels(batch["labels"], col, task)
                arr = np.array(
                    [float(v) if v is not None else np.nan for v in raw],
                    dtype=np.float32,
                )
                mask = ~np.isnan(arr)
                if not mask.any():
                    continue
                logits_buf[task].append(outputs.tensor(task).cpu().numpy().flatten()[mask])
                labels_buf[task].append(arr[mask])

    if was_training:
        model.train()

    aucs: dict[str, float] = {}
    for task in task_names:
        if not logits_buf[task] or not labels_buf[task]:
            continue
        y_true = np.concatenate(labels_buf[task])
        y_pred = np.concatenate(logits_buf[task])
        if len(y_true) > 0:
            aucs[task] = _auc(y_true, _sigmoid(y_pred))
    return aucs
