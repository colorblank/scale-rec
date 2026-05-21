"""discover-main-sort 训练脚本。

pandas chunk read 流式读取单文件 TSV，头部取验证集，每 epoch 重读训练。

5 任务 ESMM：click / cvr / detail / stock / stay
  - 分类任务 BCE，stay 用 weighted BCE: -(t/(1+t))·log p - (1/(1+t))·log(1-p)
  - P(detail|stock)=σ(click)·σ(X), P(cvr)=σ(click)·σ(cvr), P(stay)=σ(detail)·σ(stay)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.export import export_to_safetensors  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))

DEFAULT_FEATURE_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_discover.yaml")
DEFAULT_MODEL_CONFIG = os.path.join(DEMO_DIR, "model_discover_esmm.yaml")

NULL_MARKERS: set[str] = {"NULL", "\\N", "null", "None", ""}

# 模型 Batch 字典类型
Batch = dict[str, Any]  # {"features": [dict, ...], "labels": {name: [value, ...]}}

logger = logging.getLogger("train")

# ═══════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════


def _weighted_bce_stay(logits: torch.Tensor, stay_times: torch.Tensor) -> torch.Tensor:
    """Weighted BCE for stay_time task.

    p = σ(z), t = stay_time.  Loss = -(t/(1+t))·log p - (1/(1+t))·log(1-p).
    t < 0 视为缺失，不参与损失。

    Args:
        logits: [N, 1] 模型输出 logits。
        stay_times: [N, 1] 实际停留时长（秒），-1 表示缺失。

    Returns:
        标量损失值。若全部缺失则返回 0。
    """
    t = stay_times.float()
    mask = (t >= 0).squeeze(-1)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    z, t = logits[mask], t[mask]
    log_p = -F.softplus(-z)
    log_1mp = -F.softplus(z)
    return -(t / (1 + t) * log_p + 1 / (1 + t) * log_1mp).mean()


def _bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard binary cross entropy (with logits)."""
    return F.binary_cross_entropy_with_logits(logits, labels)


# ═══════════════════════════════════════════════════════════════════
# Loss computation
# ═══════════════════════════════════════════════════════════════════


def _pick_labels(batch_labels: dict[str, list[Any]], *names: str) -> list[Any]:
    """按优先级从 batch 标签中取值，跳过全为 None 的列。

    Args:
        batch_labels: 标签字典 {列名: [值, ...]}。
        *names: 列名优先级列表，先匹配到的有效列立即返回。

    Returns:
        第一个含有非 None 值的标签列，若无则返回空列表。
    """
    for n in names:
        vals = batch_labels.get(n)
        if vals and any(v is not None for v in vals):
            return vals
    return []


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list[Any]],
    label_map: dict[str, str],
) -> torch.Tensor | None:
    """计算 5 任务多任务损失。

    click / cvr / detail / stock → standard BCE
    stay → weighted BCE

    Args:
        outputs: 模型前向输出 {task_name: logits_tensor}。
        batch_labels: 批次标签字典。
        label_map: {task_name: label_column_name} 映射。

    Returns:
        多任务损失总和，若所有任务均无有效标签则返回 None。
    """
    total: torch.Tensor | None = None
    for task, logits in outputs.items():
        if task.startswith("ct"):  # 跳过 ESMM 乘积键 (ctcvr, ctdetail, ctstock, ctstay)
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
# Metrics
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
    # 每个正样本下方有多少负样本
    cum_neg = np.cumsum(1 - y)
    pos_mask = y == 1
    return float(np.sum(n_neg - cum_neg[pos_mask]) / (n_pos * n_neg))


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


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════


_DTYPE_MAP = {"int": "Int64", "float": "float64", "string": "str"}


def _parse_default(val_str: str, dtype_tag: str) -> Any:
    """按 dtype 解析配置中的 default_val。"""
    if dtype_tag == "int":
        return int(float(val_str)) if val_str else 0
    elif dtype_tag == "float":
        return float(val_str) if val_str else 0.0
    return str(val_str)


def stream_file_batches(
    path: str,
    flow_config: FlowConfig,
    batch_size: int,
    *,
    has_header: bool = True,
    sep: str = "\t",
    null_markers: set[str] | None = None,
) -> Iterator[Batch]:
    """pandas chunk read 流式读取单文件，按 role 分离 feature/label。

    列名、类型、缺失填充值全部来自 flow_config.sources 配置。

    Args:
        path: TSV 文件路径。
        flow_config: 特征配置（含 feature + label + discard source）。
        batch_size: 批大小（pandas chunksize）。
        has_header: 文件是否含 header 行。
        sep: 字段分隔符。
        null_markers: NULL 字符串集合。

    Yields:
        {"features": [row_dict, ...], "labels": {label_name: [value, ...]}}
    """
    if null_markers is None:
        null_markers = NULL_MARKERS
    na_vals = list(null_markers)

    feature_sources = flow_config.feature_sources
    label_sources = flow_config.label_sources
    discard_names = {s.name for s in flow_config.discard_sources}

    # 构建 pandas 读取 schema（按 config 原始顺序，去重列名）
    seen: set[str] = set()
    names: list[str] = []
    dtype: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    for s in flow_config.sources:
        if s.name in seen:
            continue
        seen.add(s.name)
        names.append(s.name)
        dt = s.dtype.tag if hasattr(s.dtype, "tag") else str(s.dtype)
        dtype[s.name] = _DTYPE_MAP.get(dt, "str")
        defaults[s.name] = _parse_default(s.default_val, dt)

    params: dict[str, Any] = {
        "sep": sep,
        "dtype": dtype,
        "na_values": na_vals,
        "keep_default_na": False,
        "chunksize": batch_size,
    }
    if has_header:
        params["header"] = 0
    else:
        params["header"] = None
        params["names"] = names

    source_set = {s.name for s in feature_sources}
    label_names = [s.name for s in label_sources]

    for chunk in pd.read_csv(path, **params):
        # 丢弃 discard 列
        chunk = chunk.drop(
            columns=[c for c in discard_names if c in chunk.columns], errors="ignore"
        )
        # 按配置填充缺失值
        for col, default in defaults.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].fillna(default)

        # 特征行
        rows = chunk[list(source_set & set(chunk.columns))].to_dict("records")
        rows = [{k: v for k, v in r.items() if not pd.isna(v)} for r in rows]

        # 标签
        labels = {
            ln: [None if pd.isna(v) else v for v in chunk[ln].tolist()]
            for ln in label_names
            if ln in chunk.columns
        }

        yield {"features": rows, "labels": labels}


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════


def _collect_batches(batches: Iterator[Batch], max_samples: int) -> list[Batch]:
    """从 batch 流中收集最多 max_samples 个样本。"""
    result: list[Batch] = []
    collected = 0
    for b in batches:
        result.append(b)
        collected += len(b["features"])
        if collected >= max_samples:
            break
    return result


def _to_device(d: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in d.items()}


def train_on_file(
    args: argparse.Namespace,
    flow_config: FlowConfig,
    dag: FeatureDag,
    model: torch.nn.Module,
    task_names: list[str],
    label_map: dict[str, str],
    device: torch.device,
) -> float:
    """单文件模式：流式读取，首段作验证集，每 epoch 重读文件训练。"""
    eval_samples = getattr(args, "eval_samples", 2000)
    eval_interval = getattr(args, "eval_interval", 50)
    log_interval = getattr(args, "log_interval", 10)

    # ── 第一遍：收集验证集（文件头部 eval_samples 行）──
    reader = stream_file_batches(
        args.data,
        flow_config,
        args.batch_size,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
    )
    eval_batches = _collect_batches(reader, eval_samples)
    n_eval_batches = len(eval_batches)
    eval_rows = sum(len(b["features"]) for b in eval_batches)
    logger.info("validation: %d samples (%d batches)", eval_rows, n_eval_batches)

    # ── 训练 ──
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best: float = 0.0
    ckpt_path = getattr(args, "export_path", None) or os.path.join(
        DEMO_DIR, "temp", "model.safetensors"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss: float = 0.0
        n_batches: int = 0
        t_data = t_preproc = t_forward = t_loss = t_backward = 0.0
        t0_epoch = time.perf_counter()

        stream = enumerate(
            stream_file_batches(
                args.data, flow_config, args.batch_size,
                has_header=not args.no_header, sep=args.separator,
                null_markers=set(args.null_markers),
            )
        )
        t0_iter = time.perf_counter()
        for i, batch in stream:
            t_data += time.perf_counter() - t0_iter
            if i < n_eval_batches:
                t0_iter = time.perf_counter()
                continue

            t0 = time.perf_counter()
            feat = _to_device(dag.preprocess_batch(batch["features"]), device)
            t_preproc += time.perf_counter() - t0

            t0 = time.perf_counter()
            outputs = model(feat)
            t_forward += time.perf_counter() - t0

            t0 = time.perf_counter()
            loss = compute_loss(outputs, batch["labels"], label_map)
            t_loss += time.perf_counter() - t0

            if loss is None:
                continue

            t0 = time.perf_counter()
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_backward += time.perf_counter() - t0

            total_loss += loss.item()
            n_batches += 1

            if n_batches % log_interval == 0:
                logger.info(
                    "  batch %4d  avg_loss=%.6f  cur_loss=%.6f",
                    n_batches, total_loss / n_batches, loss.item(),
                )

            if n_batches % eval_interval == 0:
                t0 = time.perf_counter()
                model.eval()
                with torch.no_grad():
                    aucs = compute_aucs(model, dag, eval_batches, task_names, label_map, device)
                model.train()
                t_eval = (time.perf_counter() - t0) * 1000
                parts = [f"batch {n_batches:4d}  loss={total_loss / n_batches:.6f}  eval={t_eval:.0f}ms"]
                for t in sorted(task_names):
                    if t in aucs and t != "stay":
                        parts.append(f"{t}: auc={aucs[t]:.4f}")
                logger.info("  " + "  ".join(parts))

            t0_iter = time.perf_counter()

        t_epoch = time.perf_counter() - t0_epoch
        avg_loss = total_loss / max(n_batches, 1)

        # epoch 结束：完整验证
        t0 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            aucs = compute_aucs(model, dag, eval_batches, task_names, label_map, device)
        t_eval = (time.perf_counter() - t0) * 1000
        _log_epoch(epoch, args.epochs, avg_loss, aucs, task_names)

        # 耗时统计
        _log_timing(epoch, t_epoch, n_batches, t_data, t_preproc, t_forward, t_loss, t_backward, t_eval)

        cur = max(aucs.values(), default=0)
        if cur > best:
            best = cur
            export_to_safetensors(model, ckpt_path)
            logger.info("checkpoint saved to %s (best auc=%.4f)", ckpt_path, best)

    return best


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def _log_epoch(
    epoch: int,
    total_epochs: int,
    loss: float,
    aucs: dict[str, float],
    task_names: list[str],
) -> None:
    """打印 epoch 训练日志（stay 不计算 AUC，不输出）。"""
    parts = [f"epoch {epoch:3d}/{total_epochs}  loss={loss:.6f}"]
    for t in sorted(task_names):
        if t in aucs and t != "stay":
            parts.append(f"{t}: auc={aucs[t]:.4f}")
    logger.info("  ".join(parts))


def _log_timing(
    epoch: int,
    t_epoch: float,
    n_batches: int,
    t_data: float,
    t_preproc: float,
    t_forward: float,
    t_loss: float,
    t_backward: float,
    t_eval: float,
) -> None:
    """打印每 epoch 各阶段耗时分布。"""
    if n_batches == 0:
        return
    t_total = t_data + t_preproc + t_forward + t_loss + t_backward
    ms = lambda s: s * 1000 / n_batches  # noqa: E731
    pct = lambda s: s / t_total * 100 if t_total > 0 else 0  # noqa: E731
    logger.info(
        "  [timing epoch %d] total=%.1fs  batches=%d  eval=%.0fms | "
        "per_batch: data=%.1fms(%.0f%%) preproc=%.1fms(%.0f%%) "
        "forward=%.1fms(%.0f%%) loss=%.1fms(%.0f%%) backward=%.1fms(%.0f%%)",
        epoch, t_epoch, n_batches, t_eval,
        ms(t_data), pct(t_data), ms(t_preproc), pct(t_preproc),
        ms(t_forward), pct(t_forward), ms(t_loss), pct(t_loss),
        ms(t_backward), pct(t_backward),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="discover-main-sort 训练")
    p.add_argument("--data", required=True, help="训练数据 TSV 路径")
    p.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="DAG 完整配置")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--export-path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-header", action="store_true", help="TSV 文件无 header 行")
    p.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    p.add_argument("--separator", default="\t")
    p.add_argument("--eval-samples", type=int, default=2000, help="验证集样本数")
    p.add_argument("--eval-interval", type=int, default=50, help="训练中每隔 N batch 验证")
    p.add_argument("--log-interval", type=int, default=10, help="每隔 N batch 打印 loss")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # DAG
    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info(
        "%d sources, %d ops → %d features", len(fc.sources), len(fc.operators), len(features)
    )

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info("device: %s", device)

    # Model
    mc = ModelConfig.from_yaml(args.model_config)
    spec: dict[str, Any] = get_output_spec(mc.type, None)
    task_names: list[str] = spec["task_names"]
    label_map: dict[str, str] = spec.get("label_col_map", {"ctr": "is_click", "cvr": "is_cvr"})
    model = mc.build(features).to(device)
    n = sum(p.numel() for p in model.parameters())
    logger.info("%s  tasks=%s  params=%s", mc.type, task_names, f"{n:,}")

    # Train
    best: float = train_on_file(args, fc, dag, model, task_names, label_map, device)
    logger.info("best AUC=%.4f", best)


if __name__ == "__main__":
    main()
