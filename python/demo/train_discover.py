from __future__ import annotations

"""discover-main-sort 统一训练脚本。

支持两种数据模式：
  1. 单文件模式（demo/测试）: --data single_file.txt
     所有列（Item+User+标签）在一个 TSV 中，Polars 全量读入。

  2. 生产模式: --user-data user.txt --item-files items1.txt,items2.txt,...
     用户行为文件 + 多日物品特征文件，流式按 item_id Join，50GB 安全处理。

标签定义：
  is_click = is_click_detail OR is_click_stock
  is_cvr  = 转赞评
  stay_time = 阅读时长（秒），上限截断 360s

CTR 损失采用 time-weighted BCE：正样本权重=log(1+stay_time)，stay_time=-1 忽略。
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

# 标签列：新名+旧名别名，确保生产/测试数据兼容
LABEL_COLUMNS = [
    "is_click",
    "is_cvr",
    "ctr",
    "cvr",
    "is_click_detail",
    "is_click_stock",
    "stay_time",
]


# ═══════════════════════════════════════════════
# 损失函数
# ═══════════════════════════════════════════════


def weighted_bce_stay(
    logits: torch.Tensor,
    stay_times: torch.Tensor,
) -> torch.Tensor:
    """Weighted Cross Entropy for stay_time prediction.

    p = σ(z), t = observed stay_time
    Loss = -(t/(1+t))·log(p) - (1/(1+t))·log(1-p)

    t < 0 视为缺失，不参与损失。

    Args:
        logits: [N, 1] 模型输出 logits。
        stay_times: [N, 1] 实际停留时长（秒），-1 表示缺失。
    """
    t = stay_times.float()
    valid = (t >= 0).squeeze(-1)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits_v = logits[valid]
    t_v = t[valid]

    log_p = -F.softplus(-logits_v)
    log_one_minus_p = -F.softplus(logits_v)

    w_pos = t_v / (1.0 + t_v)
    w_neg = 1.0 / (1.0 + t_v)

    loss = -(w_pos * log_p + w_neg * log_one_minus_p)
    return loss.mean()


def standard_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard binary cross entropy for classification tasks."""
    return F.binary_cross_entropy_with_logits(logits, labels)


# ═══════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════


def _dtype_to_polars(dt: DType) -> pl.DataType:
    tag = dt.tag if hasattr(dt, "tag") else dt
    if tag == "int":
        return pl.Int64
    elif tag == "float":
        return pl.Float64
    elif tag == "string":
        return pl.Utf8
    return pl.Utf8


def load_single_file(
    path: str,
    flow_config: FlowConfig,
    has_header: bool = True,
    separator: str = "\t",
    null_markers: set[str] | None = None,
) -> pl.DataFrame:
    """单文件模式：Polars 全量读入 TSV，按 FlowConfig schema 强制类型。"""
    if null_markers is None:
        null_markers = NULL_MARKERS

    schema: dict[str, pl.DataType] = {}
    for s in flow_config.sources:
        schema[s.name] = _dtype_to_polars(s.dtype)
    # 标签列也声明类型
    for ln in LABEL_COLUMNS:
        schema[ln] = pl.Int64

    col_names = list(schema.keys())

    if has_header:
        df = pl.read_csv(
            path,
            separator=separator,
            has_header=True,
            schema_overrides=schema,
            null_values=list(null_markers),
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
        keep = [c for c in col_names if c in df.columns]
        df = df.select(keep)
    else:
        df = pl.read_csv(
            path,
            separator=separator,
            has_header=False,
            null_values=list(null_markers),
            truncate_ragged_lines=True,
            ignore_errors=True,
        )
        avail = df.columns[: len(col_names)]
        df = df.select(avail)
        df.columns = col_names
        for cn, dt in schema.items():
            if cn in df.columns:
                try:
                    df = df.with_columns(pl.col(cn).cast(dt, strict=False))
                except Exception:
                    pass
    return df


# ═══════════════════════════════════════════════
# 标签处理
# ═══════════════════════════════════════════════


def prepare_labels(df: pl.DataFrame) -> pl.DataFrame:
    """统一标签列名，处理缺失值。

    - is_click: 优先用已有列，否则从 is_click_detail|is_click_stock 计算，再否则用 ctr
    - is_cvr: 优先用已有列，否则用 cvr 别名
    - stay_time: NULL → -1
    """
    # is_click
    if "is_click" not in df.columns:
        if "is_click_detail" in df.columns and "is_click_stock" in df.columns:
            detail = df["is_click_detail"].fill_null(0)
            stock = df["is_click_stock"].fill_null(0)
            df = df.with_columns(((detail + stock) > 0).cast(pl.Int64).alias("is_click"))
        elif "ctr" in df.columns:
            df = df.with_columns(pl.col("ctr").cast(pl.Int64).alias("is_click"))

    # is_cvr
    if "is_cvr" not in df.columns and "cvr" in df.columns:
        df = df.with_columns(pl.col("cvr").cast(pl.Int64).alias("is_cvr"))

    # stay_time
    if "stay_time" in df.columns:
        df = df.with_columns(pl.col("stay_time").fill_null(-1))
    return df


# ═══════════════════════════════════════════════
# 训练 / 评估
# ═══════════════════════════════════════════════


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def auc_score(labels: np.ndarray, probs: np.ndarray) -> float:
    order = np.argsort(probs)[::-1]
    ls = labels[order]
    n_pos = ls.sum()
    n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum = np.cumsum(ls)
    return float((cum - ls).sum() / (n_pos * n_neg))


def _get_label_vals(batch_labels: dict[str, list], *names: str) -> list:
    """从 batch labels 中按优先级取标签值，支持新旧列名别名。
    跳过全为 None 的列（列存在但无有效数据）。
    """
    for n in names:
        vals = batch_labels.get(n)
        if vals and any(v is not None for v in vals):
            return vals
    return []


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch_labels: dict[str, list],
    label_col_map: dict[str, str],
) -> torch.Tensor | None:
    """计算 5 任务 loss。

    click/cvr/detail/stock → standard BCE
    stay → weighted BCE: -(t/(1+t))·log(p) - (1/(1+t))·log(1-p)
    """
    total_loss = None
    for task_name, logits in outputs.items():
        if task_name.startswith("ct"):  # skip ESMM product keys (ctcvr, ctdetail, ctstock, ctstay)
            continue
        label_col = label_col_map.get(task_name)
        if label_col is None:
            continue
        raw = _get_label_vals(batch_labels, label_col, task_name)
        if not raw:
            continue

        arr = np.array([float(v) if v is not None else np.nan for v in raw], dtype=np.float32)
        valid = ~np.isnan(arr)
        if not valid.any():
            continue

        if task_name == "stay":
            stay_t = torch.tensor(arr[valid], dtype=torch.float32).view(-1, 1)
            task_loss = weighted_bce_stay(logits[valid], stay_t)
        else:
            labels = torch.tensor(arr[valid], dtype=torch.float32).view(-1, 1)
            task_loss = standard_bce(logits[valid], labels)

        total_loss = task_loss if total_loss is None else total_loss + task_loss
    return total_loss


def evaluate_aucs(model, dag, batches, task_names, label_col_map) -> dict[str, float]:
    model.eval()
    all_logits: dict[str, list] = {t: [] for t in task_names}
    all_labels: dict[str, list] = {t: [] for t in task_names}
    with torch.no_grad():
        for batch in batches:
            rows = [{k: v for k, v in row.items() if v is not None} for row in batch["features"]]
            tensors = dag.preprocess_batch(rows)
            outputs = model(tensors)
            for t in task_names:
                if t not in outputs:
                    continue
                all_logits[t].append(outputs[t].cpu().numpy().flatten())
                label_col = label_col_map.get(t, t)
                raw = _get_label_vals(batch["labels"], label_col, t)
                arr = np.array(
                    [float(v) if v is not None else np.nan for v in raw], dtype=np.float32
                )
                valid = ~np.isnan(arr)
                all_labels[t].append(arr[valid])
    model.train()
    aucs = {}
    for t in task_names:
        if all_logits[t] and all_labels[t]:
            logits = np.concatenate(all_logits[t])
            y = np.concatenate(all_labels[t])
            if len(y) == 0:
                aucs[t] = 0.5
                continue
            aucs[t] = auc_score(y, sigmoid(logits))
    return aucs


# ═══════════════════════════════════════════════
# 训练循环 — 单文件模式
# ═══════════════════════════════════════════════


def train_single_file(
    args,
    flow_config,
    dag,
    model,
    features,
    task_names,
    label_col_map,
):
    df = load_single_file(
        args.data,
        flow_config,
        has_header=not args.no_header,
        separator=args.separator,
        null_markers=set(args.null_markers),
    )
    df = prepare_labels(df)

    label_cols_in_df = [c for c in ["is_click", "is_cvr"] if c in df.columns]
    has_label = pl.any_horizontal([pl.col(c).is_not_null() for c in label_cols_in_df])
    df = df.filter(has_label)
    df = df.sample(fraction=1.0, seed=42)
    n_train = int(len(df) * 0.8)
    train_df = df.slice(0, n_train)
    test_df = df.slice(n_train, len(df) - n_train)
    print(f"[Data] single-file mode: train={len(train_df)} test={len(test_df)}")

    source_names = set(dag.sources.keys())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(train_df), args.batch_size):
            batch_df = train_df.slice(start, args.batch_size)
            rows = batch_df.select(list(source_names)).to_dicts()
            rows = [{k: v for k, v in row.items() if v is not None} for row in rows]
            tensors = dag.preprocess_batch(rows)
            outputs = model(tensors)

            # 构建 batch labels dict
            b_labels: dict[str, list] = {}
            for c in LABEL_COLUMNS:
                if c in batch_df.columns:
                    b_labels[c] = batch_df[c].to_list()

            loss = compute_loss(outputs, b_labels, label_col_map)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # 评估
        aucs = _eval_single_file(
            model, dag, test_df, source_names, task_names, label_col_map, args.batch_size
        )
        _log_epoch(epoch, args.epochs, total_loss / max(n_batches, 1), aucs, task_names)
        for v in aucs.values():
            if v > best_auc:
                best_auc = v

    return best_auc


def _eval_single_file(model, dag, df, source_names, task_names, label_col_map, batch_size):
    model.eval()
    all_logits: dict[str, list] = {t: [] for t in task_names}
    all_labels: dict[str, list] = {t: [] for t in task_names}
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            bdf = df.slice(start, batch_size)
            rows = bdf.select(list(source_names)).to_dicts()
            rows = [{k: v for k, v in row.items() if v is not None} for row in rows]
            outputs = model(dag.preprocess_batch(rows))
            for t in task_names:
                if t in outputs:
                    all_logits[t].append(outputs[t].cpu().numpy().flatten())
                    lc = label_col_map.get(t, t)
                    if lc in bdf.columns:
                        vals = bdf[lc].to_numpy()
                        valid = ~np.isnan(vals.astype(np.float64))
                        all_labels[t].append(vals[valid].astype(np.float32))
    model.train()
    aucs = {}
    for t in task_names:
        if all_logits[t] and all_labels[t]:
            lg = np.concatenate(all_logits[t])
            y = np.concatenate(all_labels[t])
            if len(y):
                aucs[t] = auc_score(y, sigmoid(lg))
    return aucs


# ═══════════════════════════════════════════════
# 训练循环 — 生产流式模式
# ═══════════════════════════════════════════════


def train_prod(
    args,
    flow_config,
    dag,
    model,
    features,
    task_names,
    label_col_map,
):
    item_files = [p.strip() for p in args.item_files.split(",") if p.strip()]
    item_source_names = [s.name for s in flow_config.sources if s.source == "Item"]
    all_source_names = [s.name for s in flow_config.sources]
    source_dtypes = {
        s.name: s.dtype.tag if hasattr(s.dtype, "tag") else str(s.dtype)
        for s in flow_config.sources
    }

    # 构建物品索引
    item_index = build_item_index(
        item_files,
        item_source_names,
        separator=args.separator,
        null_markers=set(args.null_markers),
    )
    print(f"[ItemIndex] {len(item_index)} unique items")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = 0.0
    has_stay = "stay_time" in LABEL_COLUMNS

    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        eval_batches, eval_count = [], 0

        for batch in stream_join(
            args.user_data,
            item_index,
            all_source_names,
            source_dtypes,
            LABEL_COLUMNS,
            batch_size=args.batch_size,
            separator=args.separator,
            null_markers=set(args.null_markers),
            skip_missing_item=args.skip_missing_item,
        ):
            if eval_count < args.eval_samples:
                eval_batches.append(batch)
                eval_count += len(batch["features"])

            loss = compute_loss(
                model(dag.preprocess_batch(batch["features"])),
                batch["labels"],
                label_col_map,
            )
            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        aucs = evaluate_aucs(model, dag, eval_batches, task_names, label_col_map)
        _log_epoch(epoch, args.epochs, total_loss / max(n_batches, 1), aucs, task_names)
        for v in aucs.values():
            if v > best_auc:
                best_auc = v

    return best_auc


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════


def _log_epoch(epoch, total_epochs, loss, aucs, task_names):
    parts = [f"epoch {epoch:3d}/{total_epochs}  loss={loss:.6f}"]
    for t in sorted(task_names):
        if t in aucs and t != "stay":  # stay 为连续值，不计算 AUC
            parts.append(f"{t}: auc={aucs[t]:.4f}")
    print("  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="discover-main-sort 统一训练脚本")
    # 数据
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data", help="单文件路径（demo 模式）")
    g.add_argument("--user-data", help="用户行为文件路径（生产模式）")
    parser.add_argument("--item-files", help="物品特征文件，逗号分隔（生产模式）")
    # 配置
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--export-path")
    # 训练
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    # 数据处理
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    parser.add_argument("--separator", default="\t")
    parser.add_argument("--skip-missing-item", action="store_true")
    parser.add_argument("--no-time-weight", action="store_true")
    parser.add_argument("--eval-samples", type=int, default=2000)
    args = parser.parse_args()

    # 生产模式校验
    if args.user_data and not args.item_files:
        parser.error("--item-files required with --user-data")

    # ── Build DAG ──
    flow_config = FlowConfig.from_yaml(args.feature_config)
    print(f"[Config] {len(flow_config.sources)} sources, {len(flow_config.operators)} ops")
    dag = FeatureDag(flow_config)
    features = dag.feature_tuples()
    print(f"[DAG] {len(features)} embeddable features")

    # ── Build model ──
    model_config = ModelConfig.from_yaml(args.model_config)
    spec = get_output_spec(model_config.type, None)
    task_names = spec.get("task_names", ["ctr", "cvr"])
    label_col_map = spec.get("label_col_map", {"ctr": "is_click", "cvr": "is_cvr"})
    print(f"[Model] type={model_config.type}, tasks={task_names}, label_map={label_col_map}")

    model = model_config.build(features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params:,}")

    # ── Train ──
    if args.user_data:
        best_auc = train_prod(
            args,
            flow_config,
            dag,
            model,
            features,
            task_names,
            label_col_map,
        )
    else:
        best_auc = train_single_file(
            args,
            flow_config,
            dag,
            model,
            features,
            task_names,
            label_col_map,
        )

    print(f"[Best] AUC={best_auc:.4f}")

    # ── Export ──
    export_path = args.export_path or os.path.join(DEMO_DIR, "temp", "model.safetensors")
    export_to_safetensors(model, export_path)
    print(f"[Export] {export_path}")
    print("Done.")


if __name__ == "__main__":
    main()
