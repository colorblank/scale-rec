"""discover-main-sort 训练脚本。

支持两种数据模式：
  1. 单文件: --data file.txt（Item+User+标签合一，Polars 全量读入）
  2. 生产流式: --user-data user.txt --item-files items1.txt,...（多日物品索引 + Join）

5 任务 ESMM：click / cvr / detail / stock / stay
  - 分类任务 BCE，stay 用 weighted BCE: -(t/(1+t))·log p - (1/(1+t))·log(1-p)
  - P(detail|stock)=σ(click)·σ(X), P(cvr)=σ(click)·σ(cvr), P(stay)=σ(detail)·σ(stay)
"""

from __future__ import annotations

import argparse
import os
import sys
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
from train.data import build_item_index, stream_join  # noqa: E402
from train.export import export_to_safetensors  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))

DEFAULT_FEATURE_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_discover.yaml")
DEFAULT_ITEM_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_item.yaml")
DEFAULT_USER_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_user.yaml")
DEFAULT_MODEL_CONFIG = os.path.join(DEMO_DIR, "model_discover_esmm.yaml")
DEFAULT_DATA = os.path.join(DEMO_DIR, "temp", "discover_train_data.txt")

NULL_MARKERS: set[str] = {"NULL", "\\N", "null", "None", ""}
LABEL_COLUMNS: list[str] = [
    "is_click",
    "is_cvr",
    "ctr",
    "cvr",
    "is_click_detail",
    "is_click_stock",
    "stay_time",
]

# 模型 Batch 字典类型
Batch = dict[str, Any]  # {"features": [dict, ...], "labels": {name: [value, ...]}}

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
            t = torch.tensor(arr[valid], dtype=torch.float32).view(-1, 1)
            loss = _weighted_bce_stay(logits[valid], t)
        else:
            y = torch.tensor(arr[valid], dtype=torch.float32).view(-1, 1)
            loss = _bce(logits[valid], y)

        total = loss if total is None else total + loss
    return total


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Fast AUC via sorting."""
    order = np.argsort(probs)[::-1]
    y = labels[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((np.cumsum(y) - y).sum() / (n_pos * n_neg))


def compute_aucs(
    model: torch.nn.Module,
    dag: FeatureDag,
    batches: list[Batch],
    task_names: list[str],
    label_map: dict[str, str],
) -> dict[str, float]:
    """在多个 batch 上计算各分类任务的 AUC。

    Args:
        model: 训练中的模型（会在 eval 模式下运行后恢复 train）。
        dag: 特征 DAG。
        batches: 评估批次列表。
        task_names: 需评估的任务名列表。
        label_map: {task_name: label_column_name} 映射。

    Returns:
        {task_name: auc_value}，仅包含有有效标签的任务。
    """
    model.eval()
    logits_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
    labels_buf: dict[str, list[np.ndarray]] = {t: [] for t in task_names}
    with torch.no_grad():
        for batch in batches:
            rows = [{k: v for k, v in r.items() if v is not None} for r in batch["features"]]
            outputs = model(dag.preprocess_batch(rows))
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


def _read_tsv(
    path: str,
    schema: dict[str, str],
    *,
    has_header: bool = True,
    sep: str = "\t",
    null_markers: set[str] = NULL_MARKERS,
) -> pd.DataFrame:
    """按 pandas dtype 映射 + null 标记读取 TSV。"""
    cols = list(schema)
    na_vals = list(null_markers)
    params: dict[str, Any] = {
        "sep": sep,
        "dtype": schema,
        "na_values": na_vals,
        "keep_default_na": False,
    }
    if has_header:
        params["header"] = 0
    else:
        params["header"] = None
        params["names"] = cols
    df = pd.read_csv(path, **params)
    return df[[c for c in cols if c in df.columns]]


def load_single_file(
    path: str,
    flow_config: FlowConfig,
    *,
    has_header: bool = True,
    sep: str = "\t",
    null_markers: set[str] = NULL_MARKERS,
) -> pd.DataFrame:
    """单文件模式：按 FlowConfig schema + LABEL_COLUMNS 类型化读入 TSV。

    dtype/default_val 全部来自配置。
    """
    # 构建 schema 和默认值
    schema: dict[str, str] = {}
    defaults: dict[str, Any] = {}
    for s in flow_config.sources:
        dt = s.dtype.tag if hasattr(s.dtype, "tag") else str(s.dtype)
        schema[s.name] = _DTYPE_MAP.get(dt, "str")
        defaults[s.name] = _parse_default(s.default_val, dt)
    for ln in LABEL_COLUMNS:
        schema[ln] = "Int64"
        defaults[ln] = 0

    df = _read_tsv(path, schema, has_header=has_header, sep=sep, null_markers=null_markers)
    # 按配置填充缺失值
    for col, default in defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df


def _parse_default(val_str: str, dtype_tag: str) -> Any:
    """按 dtype 解析配置中的 default_val。"""
    if dtype_tag == "int":
        return int(float(val_str)) if val_str else 0
    elif dtype_tag == "float":
        return float(val_str) if val_str else 0.0
    return str(val_str)


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    """统一标签列名并处理缺失值。

    - is_click: 优先已有列，否则从 is_click_detail|is_click_stock 计算，再否则用 ctr
    - is_cvr: 优先已有列，否则用 cvr 别名
    - stay_time: NULL → -1

    Args:
        df: 输入 DataFrame。

    Returns:
        处理后的 DataFrame。
    """
    if "is_click" not in df.columns:
        if "is_click_detail" in df.columns and "is_click_stock" in df.columns:
            df["is_click"] = (
                (df["is_click_detail"].fillna(0) + df["is_click_stock"].fillna(0)) > 0
            ).astype("Int64")
        elif "ctr" in df.columns:
            df["is_click"] = df["ctr"].astype("Int64")
    if "is_cvr" not in df.columns and "cvr" in df.columns:
        df["is_cvr"] = df["cvr"].astype("Int64")
    if "stay_time" in df.columns:
        df["stay_time"] = df["stay_time"].fillna(-1)
    return df


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dag: FeatureDag,
    batch_iter: Iterator[Batch],
    task_names: list[str],
    label_map: dict[str, str],
    *,
    eval_batches: list[Batch] | None = None,
    eval_max: int = 0,
) -> tuple[float, list[Batch]]:
    """通用训练 epoch，batch_iter 产出 batch dict。

    生产模式下可同时收集前 eval_max 个样本用于评估。

    Args:
        model: 模型。
        optimizer: 优化器。
        dag: 特征 DAG。
        batch_iter: batch 迭代器，每个 batch 含 "features" 和 "labels"。
        task_names: 任务名列表。
        label_map: {task_name: label_column} 映射。
        eval_batches: 用于收集评估样本的列表（生产模式复用）。
        eval_max: 最多收集的评估样本数。

    Returns:
        (平均损失, 评估批次列表)。
    """
    model.train()
    total_loss: float = 0.0
    n: int = 0
    eval_cap: list[Batch] = eval_batches if eval_batches is not None else []
    collected: int = 0

    for batch in batch_iter:
        if eval_max and collected < eval_max:
            eval_cap.append(batch)
            collected += len(batch["features"])

        loss = compute_loss(
            model(dag.preprocess_batch(batch["features"])), batch["labels"], label_map
        )
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1), eval_cap


# ── 单文件模式 ──


def _file_batches(
    df: pd.DataFrame,
    source_names: set[str],
    batch_size: int,
    label_cols: list[str],
) -> Iterator[Batch]:
    """将 DataFrame 切片为 batch 迭代器。

    Args:
        df: 全量 DataFrame。
        source_names: DAG source 列名集合。
        batch_size: 批次大小。
        label_cols: 需提取的标签列名。

    Yields:
        {"features": [row_dict, ...], "labels": {col: [val, ...]}}。
    """
    for start in range(0, len(df), batch_size):
        bdf = df.iloc[start : start + batch_size]
        rows = bdf[list(source_names)].to_dict("records")
        rows = [{k: v for k, v in r.items() if not pd.isna(v)} for r in rows]
        labels = {c: bdf[c].tolist() for c in label_cols if c in bdf.columns}
        yield {"features": rows, "labels": labels}


def train_on_file(
    args: argparse.Namespace,
    flow_config: FlowConfig,
    dag: FeatureDag,
    model: torch.nn.Module,
    task_names: list[str],
    label_map: dict[str, str],
) -> float:
    """单文件模式训练。

    Args:
        args: 命令行参数。
        flow_config: 特征配置。
        dag: 特征 DAG。
        model: 模型。
        task_names: 任务名列表。
        label_map: {task_name: label_column} 映射。

    Returns:
        最佳 AUC 值。
    """
    df = load_single_file(
        args.data,
        flow_config,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
    )
    df = prepare_labels(df)
    label_checks = [c for c in ["is_click", "is_cvr"] if c in df.columns]
    mask = pd.concat([df[c].notna() for c in label_checks], axis=1).any(axis=1)
    df = df[mask]
    df = df.sample(frac=1.0, random_state=42)
    n = int(len(df) * 0.8)
    train_df = df.iloc[:n]
    test_df = df.iloc[n:]
    print(f"[Data] single-file: train={len(train_df)} test={len(test_df)}")

    source_set = set(dag.sources.keys())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best: float = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, _ = train_epoch(
            model,
            opt,
            dag,
            _file_batches(train_df, source_set, args.batch_size, LABEL_COLUMNS),
            task_names,
            label_map,
        )
        aucs = compute_aucs(
            model,
            dag,
            list(_file_batches(test_df, source_set, args.batch_size, LABEL_COLUMNS)),
            task_names,
            label_map,
        )
        _log_epoch(epoch, args.epochs, avg_loss, aucs, task_names)
        best = max(best, max(aucs.values(), default=0))

    return best


# ── 生产流式模式 ──


def train_on_prod(
    args: argparse.Namespace,
    flow_config: FlowConfig,
    dag: FeatureDag,
    model: torch.nn.Module,
    task_names: list[str],
    label_map: dict[str, str],
) -> float:
    """生产流式模式训练：多日物品索引 + 流式 Join。

    Args:
        args: 命令行参数（需含 user_data, item_files）。
        flow_config: 特征配置。
        dag: 特征 DAG。
        model: 模型。
        task_names: 任务名列表。
        label_map: {task_name: label_column} 映射。

    Returns:
        最佳 AUC 值。
    """
    item_files = [p.strip() for p in args.item_files.split(",") if p.strip()]

    # 从 YAML 配置文件读取 source 定义（name, dtype, default_val 全部来自配置）
    import yaml

    with open(args.item_config) as f:
        item_sources: list[dict] = yaml.safe_load(f)["sources"]
    with open(args.user_config) as f:
        user_sources: list[dict] = yaml.safe_load(f)["sources"]

    # 分离 source 列和 label 列
    all_sources = [
        s for s in user_sources if s["name"] in {src.name for src in flow_config.sources}
    ]
    label_sources = [
        s for s in user_sources if s["name"] not in {src.name for src in flow_config.sources}
    ]

    item_idx = build_item_index(
        item_files,
        item_sources,
        separator=args.separator,
        has_header=not args.item_no_header,
        null_markers=set(args.null_markers),
    )
    print(f"[ItemIndex] {len(item_idx)} items from {len(item_files)} files")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best: float = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, eval_batches = train_epoch(
            model,
            opt,
            dag,
            stream_join(
                args.user_data,
                item_idx,
                all_sources,
                label_sources,
                batch_size=args.batch_size,
                separator=args.separator,
                has_header=not args.no_header,
                null_markers=set(args.null_markers),
                skip_missing_item=args.skip_missing_item,
            ),
            task_names,
            label_map,
            eval_batches=[],
            eval_max=args.eval_samples,
        )
        aucs = compute_aucs(model, dag, eval_batches, task_names, label_map)
        _log_epoch(epoch, args.epochs, avg_loss, aucs, task_names)
        best = max(best, max(aucs.values(), default=0))

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
    print("  " + "  ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser(description="discover-main-sort 训练")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data")
    g.add_argument("--user-data")
    p.add_argument("--item-files")
    p.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG, help="DAG 完整配置")
    p.add_argument("--item-config", default=DEFAULT_ITEM_CONFIG, help="Item 侧读取配置")
    p.add_argument("--user-config", default=DEFAULT_USER_CONFIG, help="User 侧读取配置")
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--export-path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-header", action="store_true", help="用户行为文件无 header")
    p.add_argument("--item-no-header", action="store_true", help="物品文件无 header")
    p.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    p.add_argument("--separator", default="\t")
    p.add_argument("--skip-missing-item", action="store_true")
    p.add_argument("--eval-samples", type=int, default=2000)
    args = p.parse_args()

    if args.user_data and not args.item_files:
        p.error("--item-files required with --user-data")

    # DAG
    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    print(f"[Config] {len(fc.sources)} sources, {len(fc.operators)} ops → {len(features)} features")

    # Model
    mc = ModelConfig.from_yaml(args.model_config)
    spec: dict[str, Any] = get_output_spec(mc.type, None)
    task_names: list[str] = spec["task_names"]
    label_map: dict[str, str] = spec.get("label_col_map", {"ctr": "is_click", "cvr": "is_cvr"})
    model = mc.build(features)
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] {mc.type}  tasks={task_names}  params={n:,}")

    # Train
    best: float = (
        train_on_prod(args, fc, dag, model, task_names, label_map)
        if args.user_data
        else train_on_file(args, fc, dag, model, task_names, label_map)
    )
    print(f"[Best] AUC={best:.4f}")

    # Export
    path = args.export_path or os.path.join(DEMO_DIR, "temp", "model.safetensors")
    export_to_safetensors(model, path)
    print(f"[Export] {path}\nDone.")


if __name__ == "__main__":
    main()
