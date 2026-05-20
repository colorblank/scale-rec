from __future__ import annotations

"""生产训练脚本：多日物品索引 + 流式 Join + DAG 预处理 + ESMM 训练。

用法:
  uv run python python/demo/train_prod.py \
    --user-data data/user_20260331.txt \
    --item-files data/items/20260325.txt,data/items/20260326.txt,...,data/items/20260331.txt \
    --feature-config examples/feature_config_discover.yaml \
    --model-config python/demo/model_discover_esmm.yaml \
    --epochs 10 --batch-size 1024
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
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

NULL_MARKERS = {"NULL", "\\N", "null", "None", ""}
LABEL_NAMES = ["ctr", "cvr", "is_click_detail", "is_click_stock"]


# ── 训练工具 ──


def train_step(model, optimizer, dag, batch, task_names, label_col_map):
    """单个 batch 的训练，处理 NULL 标签和缺失 item 特征。"""
    feature_rows = batch["features"]
    labels_raw = batch["labels"]

    # 过滤值为 None 的 key（DAG 用 parse_default 回填）
    feature_rows = [
        {k: v for k, v in row.items() if v is not None} for row in feature_rows
    ]

    tensors = dag.preprocess_batch(feature_rows)
    outputs = model(tensors)

    loss = None
    for task_name, logits in outputs.items():
        label_col = label_col_map.get(task_name)
        if label_col is None:
            continue
        label_vals = labels_raw.get(label_col, [])
        if not label_vals:
            continue

        # 转为 float tensor，NULL 替换为 NaN
        arr = np.array([float(v) if v is not None else np.nan for v in label_vals],
                       dtype=np.float32)
        valid = ~np.isnan(arr)
        if not valid.any():
            continue

        labels = torch.tensor(arr[valid], dtype=torch.float32).view(-1, 1)
        task_loss = F.binary_cross_entropy_with_logits(logits[valid], labels)
        loss = task_loss if loss is None else loss + task_loss

    if loss is None:
        return None

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def eval_auc(model, dag, batches, task_name, label_col_map):
    """在多个 batch 上计算 AUC。"""
    label_col = label_col_map.get(task_name, task_name)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            feature_rows = [
                {k: v for k, v in row.items() if v is not None}
                for row in batch["features"]
            ]
            tensors = dag.preprocess_batch(feature_rows)
            outputs = model(tensors)
            if task_name in outputs:
                logits = outputs[task_name].cpu().numpy().flatten()
                label_vals = batch["labels"].get(label_col, [])
                arr = np.array([float(v) if v is not None else np.nan for v in label_vals],
                               dtype=np.float32)
                valid = ~np.isnan(arr)
                all_logits.append(logits[valid])
                all_labels.append(arr[valid])
    model.train()
    if not all_logits:
        return 0.5
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_labels)
    if len(y) == 0:
        return 0.5
    return _auc(y, 1.0 / (1.0 + np.exp(-logits)))


def _auc(labels: np.ndarray, probs: np.ndarray) -> float:
    order = np.argsort(probs)[::-1]
    labels_sorted = labels[order]
    n_pos = labels_sorted.sum()
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    cum_pos = np.cumsum(labels_sorted)
    return float((cum_pos - labels_sorted).sum() / (n_pos * n_neg))


# ── 主流程 ──


def main():
    parser = argparse.ArgumentParser(description="生产训练：多日物品特征 + 流式 Join")
    parser.add_argument("--user-data", required=True, help="用户行为文件路径 (~50GB)")
    parser.add_argument("--item-files", required=True,
                        help="物品特征文件列表，逗号分隔，按日期从旧到新排列")
    parser.add_argument("--feature-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--export-path", default=None,
                        help="safetensors 导出路径")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-samples", type=int, default=2000,
                        help="评估用样本数（从流中截取，避免全量扫描）")
    parser.add_argument("--skip-missing-item", action="store_true",
                        help="跳过 item_id 不在索引中的行")
    parser.add_argument("--separator", default="\t")
    args = parser.parse_args()

    item_files = [p.strip() for p in args.item_files.split(",") if p.strip()]

    # ── 1. 构建 DAG ──
    flow_config = FlowConfig.from_yaml(args.feature_config)
    print(f"[Config] {len(flow_config.sources)} sources, {len(flow_config.operators)} operators")
    dag = FeatureDag(flow_config)
    features = dag.feature_tuples()
    print(f"[DAG] {len(features)} embeddable features")

    # 分类 source name
    item_source_names = [s.name for s in flow_config.sources if s.source == "Item"]
    all_source_names = [s.name for s in flow_config.sources]
    print(f"[Sources] {len(item_source_names)} Item + "
          f"{len(all_source_names) - len(item_source_names)} User/Context")

    # ── 2. 构建物品特征索引 ──
    item_index = build_item_index(
        item_files, item_source_names,
        separator=args.separator, null_markers=NULL_MARKERS,
    )
    print(f"[ItemIndex] {len(item_index)} unique items from {len(item_files)} files")

    # ── 3. 构建模型 ──
    model_config = ModelConfig.from_yaml(args.model_config)
    spec = get_output_spec(model_config.type, None)
    task_names = spec.get("task_names", ["ctr", "cvr"])
    label_col_map = spec.get("label_col_map", {})
    print(f"[Model] type={model_config.type}, tasks={task_names}")

    model = model_config.build(features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── 4. 流式训练 ──
    source_dtypes = {s.name: s.dtype.tag if hasattr(s.dtype, 'tag') else str(s.dtype)
                     for s in flow_config.sources}

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        eval_batches = []
        eval_collected = 0

        for batch in stream_join(
            args.user_data, item_index, all_source_names, source_dtypes, LABEL_NAMES,
            batch_size=args.batch_size, separator=args.separator,
            null_markers=NULL_MARKERS, skip_missing_item=args.skip_missing_item,
        ):
            # 收集评估样本（前 N 个 batch）
            if eval_collected < args.eval_samples:
                eval_batches.append(batch)
                eval_collected += len(batch["features"])

            loss_val = train_step(model, optimizer, dag, batch, task_names, label_col_map)
            if loss_val is not None:
                total_loss += loss_val
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # 评估
        parts = [f"epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  batches={n_batches}"]
        for t in task_names:
            auc = eval_auc(model, dag, eval_batches, t, label_col_map)
            if auc > best_auc:
                best_auc = auc
            parts.append(f"{t}: auc={auc:.4f}")
        print("  " + "  ".join(parts))

    print(f"[Best] AUC={best_auc:.4f}")

    # ── 5. 导出 ──
    export_path = args.export_path or os.path.join(
        DEMO_DIR, "temp", "model_prod_esmm.safetensors"
    )
    export_to_safetensors(model, export_path)
    print(f"[Export] {export_path}")
    print("Done.")


if __name__ == "__main__":
    main()
