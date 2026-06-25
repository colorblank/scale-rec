from __future__ import annotations

"""AUC / PR-AUC / GAUC 计算。"""
import numpy as np


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


def _prauc(labels: np.ndarray, probs: np.ndarray) -> float:
    """PR-AUC as average precision, grouping samples with tied scores."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0

    order = np.argsort(probs, kind="stable")[::-1]
    y = labels[order]
    scores = probs[order]
    threshold_ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    true_positives = np.cumsum(y)[threshold_ends]
    false_positives = 1 + threshold_ends - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / n_pos
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(precision * recall_delta))


def gauc(labels: np.ndarray, probs: np.ndarray, group_ids: np.ndarray) -> float:
    """Group AUC: 按 group 分别计算 AUC，样本数加权平均。"""
    if len(group_ids) == 0:
        return 0.5
    total_w, w_sum = 0.0, 0
    for g in np.unique(group_ids):
        mask = group_ids == g
        y, p = labels[mask], probs[mask]
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        w = len(y)
        total_w += _auc(y, p) * w
        w_sum += w
    return total_w / w_sum if w_sum > 0 else 0.5
