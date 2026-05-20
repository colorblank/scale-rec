from __future__ import annotations

"""discover-main-sort 训练脚本：ESMM 多任务 CTR+CVR 训练 → 导出 safetensors。"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig, DType  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.export import export_to_safetensors  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))
FEATURE_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_discover.yaml")
DATA_PATH = os.path.join(DEMO_DIR, "temp", "discover_train_data.txt")
MODEL_CONFIG = os.path.join(DEMO_DIR, "model_discover_esmm.yaml")

# ── 标签列定义（不在 FlowConfig.sources 中，需单独声明）──
LABEL_COLUMNS: dict[str, pl.DataType] = {
    "ctr": pl.Int64,
    "cvr": pl.Int64,
    "is_click_detail": pl.Int64,
    "is_click_stock": pl.Int64,
}

# NULL 字符串表示（生产数据中缺失值常以这些字符串出现）
NULL_MARKERS = {"NULL", "\\N", "null", "None", ""}


def _dtype_to_polars(dt: DType) -> pl.DataType:
    """将 FlowConfig DType 映射为 Polars 类型。"""
    tag = dt.tag if hasattr(dt, "tag") else dt
    if tag == "int":
        return pl.Int64
    elif tag == "float":
        return pl.Float64
    elif tag == "string":
        return pl.Utf8
    elif tag == "list":
        return pl.Utf8  # list 类型以 JSON/分隔字符串存储
    return pl.Utf8


def build_schema(flow_config: FlowConfig, has_header: bool) -> dict[str, pl.DataType]:
    """从 FlowConfig.sources + LABEL_COLUMNS 构建 Polars schema。

    无论 TSV 是否有 header，都按 name → dtype 映射读取，确保类型正确。
    """
    schema: dict[str, pl.DataType] = {}
    for s in flow_config.sources:
        schema[s.name] = _dtype_to_polars(s.dtype)
    for label_name, label_dtype in LABEL_COLUMNS.items():
        schema[label_name] = label_dtype
    return schema


def load_data(
    path: str,
    flow_config: FlowConfig,
    has_header: bool = True,
    separator: str = "\t",
    null_markers: set[str] | None = None,
) -> pl.DataFrame:
    """加载训练数据，处理无 header TSV、NULL 值、混合类型。

    读取流程：
    1. 从 FlowConfig.sources + LABEL_COLUMNS 构建 schema
    2. 有 header → 按名读取 + schema_overrides 强制类型
       无 header → 全列按 Utf8 读入，再按位置重命名 schema 列，其余丢弃
    3. 将 NULL 标记字符串替换为 Python None
    4. DAG 处理前过滤掉值为 None 的 key（让 DAG 用 parse_default 填默认值）

    Args:
        path: 数据文件路径。
        flow_config: 特征编排配置。
        has_header: TSV 是否含 header 行。
        separator: 字段分隔符。
        null_markers: NULL 字符串集合。

    Returns:
        DataFrame，仅含 schema 列 + 标签列。
    """
    if null_markers is None:
        null_markers = NULL_MARKERS

    schema = build_schema(flow_config, has_header)
    column_names = list(schema.keys())
    n_expected = len(column_names)

    if has_header:
        # 有 header：Polars 按名读取，schema_overrides 强制类型
        df = pl.read_csv(
            path,
            separator=separator,
            has_header=True,
            schema_overrides=schema,
            null_values=list(null_markers),
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
        # 只保留 schema 中定义的列（忽略数据中的额外列如 const_dummy）
        keep_cols = [c for c in column_names if c in df.columns]
        df = df.select(keep_cols)
    else:
        # 无 header：全列读为 Utf8，按位置取前 n_expected 列
        df = pl.read_csv(
            path,
            separator=separator,
            has_header=False,
            null_values=list(null_markers),
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
        # 取前 n_expected 列，多余列（const_dummy 等）丢弃
        available = df.columns[:n_expected]
        df = df.select(available)
        df.columns = column_names
        # 按 schema 强转类型
        for col_name, dtype in schema.items():
            if col_name in df.columns:
                try:
                    df = df.with_columns(pl.col(col_name).cast(dtype, strict=False))
                except Exception:
                    pass

    return df


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def auc_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Fast AUC."""
    order = np.argsort(probs)[::-1]
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_pos = np.cumsum(labels_sorted)
    return float((cum_pos - labels_sorted).sum() / (n_pos * n_neg))


def evaluate(model, dag, df, task_names: list[str], batch_size: int, label_col_map: dict) -> dict:
    model.eval()
    all_logits: dict[str, list] = {t: [] for t in task_names}
    all_labels: dict[str, list] = {t: [] for t in task_names}
    source_names = set(dag.sources.keys())
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.slice(start, batch_size)
            # 只传 source 列进入 DAG，标签列不进预处理
            feature_rows = batch_df.select(list(source_names)).to_dicts()
            # 将 None 替换为缺失 key（DAG 用 parse_default 填充）
            feature_rows = [
                {k: v for k, v in row.items() if v is not None} for row in feature_rows
            ]
            tensors = dag.preprocess_batch(feature_rows)
            outputs = model(tensors)
            for t in task_names:
                if t in outputs:
                    all_logits[t].append(outputs[t].cpu().numpy().flatten())
                    label_col = label_col_map.get(t, t)
                    if label_col in batch_df.columns:
                        vals = batch_df[label_col].to_numpy()
                        # 过滤 NULL 标签（不参与评估）
                        mask = ~np.isnan(vals.astype(np.float64))
                        all_labels[t].append(vals[mask].astype(np.float32))
    metrics = {}
    for t in task_names:
        if all_logits[t] and all_labels[t]:
            logits = np.concatenate(all_logits[t])
            y = np.concatenate(all_labels[t])
            if len(y) == 0:
                continue
            p = sigmoid(logits)
            acc = float(((p > 0.5).astype(np.float32) == y).mean())
            metrics[t] = {"auc": auc_score(y, p), "accuracy": acc}
    return metrics


def train_epoch(model, optimizer, dag, df, task_names, batch_size, label_col_map) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    source_names = set(dag.sources.keys())
    for start in range(0, len(df), batch_size):
        batch_df = df.slice(start, batch_size)
        actual_bs = len(batch_df)
        # 只传 source 列，过滤 None
        feature_rows = batch_df.select(list(source_names)).to_dicts()
        feature_rows = [
            {k: v for k, v in row.items() if v is not None} for row in feature_rows
        ]
        tensors = dag.preprocess_batch(feature_rows)
        outputs = model(tensors)
        loss = None
        for task_name, logits in outputs.items():
            label_col = label_col_map.get(task_name)
            if label_col is None or label_col not in batch_df.columns:
                continue
            label_vals = batch_df[label_col].to_numpy()
            # 过滤 NULL 标签
            valid = ~np.isnan(label_vals.astype(np.float64))
            if not valid.any():
                continue
            labels = torch.tensor(
                label_vals[valid], dtype=torch.float32
            ).view(-1, 1)
            task_loss = F.binary_cross_entropy_with_logits(
                logits[valid], labels
            )
            loss = task_loss if loss is None else loss + task_loss
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-config", default=FEATURE_CONFIG)
    parser.add_argument("--model-config", default=MODEL_CONFIG)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-header", action="store_true",
                        help="TSV 无 header 行时使用 FlowConfig.sources 列名")
    parser.add_argument("--null-markers", nargs="*",
                        default=list(NULL_MARKERS),
                        help="NULL 字符串标记列表")
    parser.add_argument("--separator", default="\t", help="字段分隔符")
    args = parser.parse_args()

    # ── Build DAG ──
    flow_config = FlowConfig.from_yaml(args.feature_config)
    print(f"[Config] {len(flow_config.sources)} sources, {len(flow_config.operators)} operators")
    dag = FeatureDag(flow_config)
    features = dag.feature_tuples()
    print(f"[DAG] {len(features)} embeddable features")

    # ── Load data ──
    df = load_data(
        args.data,
        flow_config,
        has_header=not args.no_header,
        separator=args.separator,
        null_markers=set(args.null_markers),
    )
    # 过滤：至少有一个 label 不为 NULL 的样本才参与训练
    label_cols = [c for c in LABEL_COLUMNS if c in df.columns]
    has_label = pl.any_horizontal(
        [pl.col(c).is_not_null() for c in label_cols]
    )
    df = df.filter(has_label)
    df = df.sample(fraction=1.0, seed=42)
    n_train = int(len(df) * 0.8)
    train_df = df.slice(0, n_train)
    test_df = df.slice(n_train, len(df) - n_train)
    print(f"[Data] train={len(train_df)} test={len(test_df)} "
          f"(header={not args.no_header}, null_markers={args.null_markers})")

    # ── Build model ──
    model_config = ModelConfig.from_yaml(args.model_config)
    print(f"[Model] type={model_config.type}")

    spec = get_output_spec(model_config.type, None)
    task_names = spec.get("task_names", ["ctr", "cvr"])
    label_col_map = spec.get("label_col_map", {})
    print(f"[Model] tasks={task_names}, label_map={label_col_map}")

    model = model_config.build(features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params:,}")

    # ── Train ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, optimizer, dag, train_df, task_names, args.batch_size, label_col_map
        )
        metrics = evaluate(model, dag, test_df, task_names, args.batch_size, label_col_map)

        parts = [f"epoch {epoch:3d}/{args.epochs}  loss={train_loss:.6f}"]
        for t, m in metrics.items():
            if m["auc"] > best_auc:
                best_auc = m["auc"]
            parts.append(f"{t}: auc={m['auc']:.4f} acc={m['accuracy']:.4f}")
        print("  " + "  ".join(parts))

    print(f"[Best] AUC={best_auc:.4f}")

    # ── Export ──
    export_path = os.path.join(DEMO_DIR, "temp", "model_discover_esmm.safetensors")
    export_to_safetensors(model, export_path)
    print(f"[Export] {export_path}")

    # ── Inference demo ──
    print("\n[Inference Demo] running on first 5 test samples (no labels)...")
    demo_df = test_df.slice(0, 5).drop([c for c in label_cols if c in test_df.columns])
    model.eval()
    source_names = set(dag.sources.keys())
    with torch.no_grad():
        feature_rows = demo_df.select(list(source_names)).to_dicts()
        feature_rows = [
            {k: v for k, v in row.items() if v is not None} for row in feature_rows
        ]
        tensors = dag.preprocess_batch(feature_rows)
        preds = model(tensors)
        for task in task_names:
            if task in preds:
                logits = preds[task].cpu().numpy().flatten()
                probs = sigmoid(logits)
                print(f"  {task}: logits={logits.round(4)} probs={probs.round(4)}")

    print("Done.")


if __name__ == "__main__":
    main()
