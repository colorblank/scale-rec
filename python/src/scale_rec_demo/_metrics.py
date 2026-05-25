from __future__ import annotations

"""共享评估指标函数（logloss, AUC, accuracy）。"""
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order]
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    u_stat = pos_ranks.sum() - n_pos * (n_pos + 1) / 2
    return float(1.0 - u_stat / (n_pos * n_neg))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((y_pred >= threshold) == (y_true == 1)))
