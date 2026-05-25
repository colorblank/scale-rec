from __future__ import annotations

"""分类指标。"""
import numpy as np


def logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy (log loss). y_pred 应为概率值 [0,1]."""
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    pred = (y_pred >= threshold).astype(np.int32)
    return float((pred == y_true).mean())


def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    pred = (y_pred >= threshold).astype(np.int32)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    pred = (y_pred >= threshold).astype(np.int32)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
