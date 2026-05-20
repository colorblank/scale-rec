from __future__ import annotations

"""discover-main-sort 训练脚本。

支持两种数据模式：
  1. 单文件: --data file.txt（Item+User+标签合一，Polars 全量读入）
  2. 生产流式: --user-data user.txt --item-files items1.txt,...（多日物品索引 + Join）

5 任务 ESMM：click / cvr / detail / stock / stay
  - 分类任务 BCE，stay 用 weighted BCE: -(t/(1+t))·log p - (1/(1+t))·log(1-p)
  - P(detail|stock)=σ(click)·σ(X), P(cvr)=σ(click)·σ(cvr), P(stay)=σ(detail)·σ(stay)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig, DType  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.data import build_item_index, stream_join  # noqa: E402
from train.export import export_to_safetensors  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))

DEFAULT_FEATURE_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_discover.yaml")
DEFAULT_MODEL_CONFIG = os.path.join(DEMO_DIR, "model_discover_esmm.yaml")
DEFAULT_DATA = os.path.join(DEMO_DIR, "temp", "discover_train_data.txt")

NULL_MARKERS = {"NULL", "\\N", "null", "None", ""}
LABEL_COLUMNS = ["is_click", "is_cvr", "ctr", "cvr",
                 "is_click_detail", "is_click_stock", "stay_time"]

# ═══════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════

def _weighted_bce_stay(logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """stay_time weighted BCE: -(t/(1+t))·log σ(z) - (1/(1+t))·log(1-σ(z)).  t<0 忽略."""
    t = t.float()
    mask = (t >= 0).squeeze(-1)
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    z, t = logits[mask], t[mask]
    log_p = -F.softplus(-z)
    log_1mp = -F.softplus(z)
    return -(t / (1 + t) * log_p + 1 / (1 + t) * log_1mp).mean()


def _bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


# ═══════════════════════════════════════════════════════════════════
# Loss computation
# ═══════════════════════════════════════════════════════════════════

def _pick_labels(batch: dict[str, list], *names: str) -> list:
    """按优先级取标签值，跳过全 None 列."""
    for n in names:
        vals = batch.get(n)
        if vals and any(v is not None for v in vals):
            return vals
    return []


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list],
    label_map: dict[str, str],
) -> torch.Tensor | None:
    """5 任务 loss：click/cvr/detail/stock → BCE, stay → weighted BCE."""
    total = None
    for task, logits in outputs.items():
        if task.startswith("ct"):  # skip ESMM products (ctcvr, ctdetail, ctstock, ctstay)
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
    batches: list[dict],
    task_names: list[str],
    label_map: dict[str, str],
) -> dict[str, float]:
    """在多个 batch 上计算分类任务 AUC."""
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
                arr = np.array([float(v) if v is not None else np.nan for v in raw], dtype=np.float32)
                labels_buf[t].append(arr[~np.isnan(arr)])
    model.train()
    aucs = {}
    for t in task_names:
        if logits_buf[t] and labels_buf[t]:
            y = np.concatenate(labels_buf[t])
            if len(y):
                aucs[t] = _auc(y, _sigmoid(np.concatenate(logits_buf[t])))
    return aucs


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def _dtype_to_polars(dt: DType) -> pl.DataType:
    t = dt.tag if hasattr(dt, "tag") else dt
    return {"int": pl.Int64, "float": pl.Float64, "string": pl.Utf8}.get(t, pl.Utf8)


def load_single_file(path: str, flow_config: FlowConfig, *,
                     has_header: bool = True, sep: str = "\t",
                     null_markers: set[str] = NULL_MARKERS) -> pl.DataFrame:
    """单文件模式：按 FlowConfig schema + LABEL_COLUMNS 类型化读入 TSV."""
    schema = {s.name: _dtype_to_polars(s.dtype) for s in flow_config.sources}
    schema.update({ln: pl.Int64 for ln in LABEL_COLUMNS})
    cols = list(schema)

    if has_header:
        df = pl.read_csv(path, separator=sep, has_header=True,
                         schema_overrides=schema, null_values=list(null_markers),
                         truncate_ragged_lines=True, ignore_errors=True)
        df = df.select([c for c in cols if c in df.columns])
    else:
        df = pl.read_csv(path, separator=sep, has_header=False,
                         null_values=list(null_markers),
                         truncate_ragged_lines=True, ignore_errors=True)
        df = df.select(df.columns[:len(cols)])
        df.columns = cols
        for cn, dt in schema.items():
            if cn in df.columns:
                try:
                    df = df.with_columns(pl.col(cn).cast(dt, strict=False))
                except Exception:
                    pass
    return df


def prepare_labels(df: pl.DataFrame) -> pl.DataFrame:
    """统一标签: is_click, is_cvr, stay_time(NULL→-1)."""
    if "is_click" not in df.columns:
        if "is_click_detail" in df.columns and "is_click_stock" in df.columns:
            df = df.with_columns(
                ((df["is_click_detail"].fill_null(0) + df["is_click_stock"].fill_null(0)) > 0)
                .cast(pl.Int64).alias("is_click"))
        elif "ctr" in df.columns:
            df = df.with_columns(pl.col("ctr").cast(pl.Int64).alias("is_click"))
    if "is_cvr" not in df.columns and "cvr" in df.columns:
        df = df.with_columns(pl.col("cvr").cast(pl.Int64).alias("is_cvr"))
    if "stay_time" in df.columns:
        df = df.with_columns(pl.col("stay_time").fill_null(-1))
    return df


# ═══════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, optimizer, dag, batch_iter, task_names, label_map,
                eval_batches: list | None = None, eval_max: int = 0) -> tuple[float, list]:
    """通用训练 epoch，batch_iter 是生成 batch dict 的迭代器."""
    model.train()
    total_loss, n = 0.0, 0
    eval_cap = []
    collected = 0

    for batch in batch_iter:
        if eval_max and collected < eval_max:
            eval_cap.append(batch)
            collected += len(batch["features"])

        loss = compute_loss(model(dag.preprocess_batch(batch["features"])),
                            batch["labels"], label_map)
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1), eval_cap


# ── 单文件模式 ──

def _file_batches(df, source_names, batch_size, label_cols):
    for start in range(0, len(df), batch_size):
        bdf = df.slice(start, batch_size)
        rows = bdf.select(list(source_names)).to_dicts()
        rows = [{k: v for k, v in r.items() if v is not None} for r in rows]
        labels = {c: bdf[c].to_list() for c in label_cols if c in bdf.columns}
        yield {"features": rows, "labels": labels}


def train_on_file(args, flow_config, dag, model, task_names, label_map):
    df = load_single_file(args.data, flow_config, has_header=not args.no_header,
                          sep=args.separator, null_markers=set(args.null_markers))
    df = prepare_labels(df)
    label_checks = [c for c in ["is_click", "is_cvr"] if c in df.columns]
    df = df.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in label_checks]))
    df = df.sample(fraction=1.0, seed=42)
    train_df = df.slice(0, int(len(df) * 0.8))
    test_df = df.slice(int(len(df) * 0.8), len(df))
    print(f"[Data] single-file: train={len(train_df)} test={len(test_df)}")

    source_set = set(dag.sources.keys())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, _ = train_epoch(model, opt, dag,
                                  _file_batches(train_df, source_set, args.batch_size, LABEL_COLUMNS),
                                  task_names, label_map)

        # eval on test set
        aucs = compute_aucs(model, dag,
                            list(_file_batches(test_df, source_set, args.batch_size, LABEL_COLUMNS)),
                            task_names, label_map)
        _log_epoch(epoch, args.epochs, avg_loss, aucs, task_names)
        best = max(best, max(aucs.values(), default=0))

    return best


# ── 生产流式模式 ──

def train_on_prod(args, flow_config, dag, model, task_names, label_map):
    item_files = [p.strip() for p in args.item_files.split(",") if p.strip()]
    item_src_names = [s.name for s in flow_config.sources if s.source == "Item"]
    all_src_names = [s.name for s in flow_config.sources]
    src_dtypes = {s.name: (s.dtype.tag if hasattr(s.dtype, "tag") else str(s.dtype))
                  for s in flow_config.sources}

    item_idx = build_item_index(item_files, item_src_names,
                                separator=args.separator, null_markers=set(args.null_markers))
    print(f"[ItemIndex] {len(item_idx)} items from {len(item_files)} files")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        avg_loss, eval_batches = train_epoch(
            model, opt, dag,
            stream_join(args.user_data, item_idx, all_src_names, src_dtypes,
                        LABEL_COLUMNS, batch_size=args.batch_size,
                        separator=args.separator, null_markers=set(args.null_markers),
                        skip_missing_item=args.skip_missing_item),
            task_names, label_map,
            eval_batches=[], eval_max=args.eval_samples,
        )
        aucs = compute_aucs(model, dag, eval_batches, task_names, label_map)
        _log_epoch(epoch, args.epochs, avg_loss, aucs, task_names)
        best = max(best, max(aucs.values(), default=0))

    return best


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def _log_epoch(epoch, total, loss, aucs, task_names):
    parts = [f"epoch {epoch:3d}/{total}  loss={loss:.6f}"]
    for t in sorted(task_names):
        if t in aucs and t != "stay":
            parts.append(f"{t}: auc={aucs[t]:.4f}")
    print("  " + "  ".join(parts))


def main():
    p = argparse.ArgumentParser(description="discover-main-sort 训练")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--data")
    g.add_argument("--user-data")
    p.add_argument("--item-files")
    p.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--export-path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-header", action="store_true")
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
    spec = get_output_spec(mc.type, None)
    task_names = spec["task_names"]
    label_map = spec.get("label_col_map", {"ctr": "is_click", "cvr": "is_cvr"})
    model = mc.build(features)
    n = sum(p.numel() for p in model.parameters())
    print(f"[Model] {mc.type}  tasks={task_names}  params={n:,}")

    # Train
    best = train_on_prod(args, fc, dag, model, task_names, label_map) if args.user_data \
        else train_on_file(args, fc, dag, model, task_names, label_map)
    print(f"[Best] AUC={best:.4f}")

    # Export
    path = args.export_path or os.path.join(DEMO_DIR, "temp", "model.safetensors")
    export_to_safetensors(model, path)
    print(f"[Export] {path}\nDone.")


if __name__ == "__main__":
    main()
