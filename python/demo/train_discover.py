from __future__ import annotations

"""discover-main-sort 训练脚本：ESMM 多任务 CTR+CVR 训练 → 导出 safetensors。"""

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

from train.config import FlowConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.export import export_to_safetensors  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))
FEATURE_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_discover.yaml")
DATA_PATH = os.path.join(DEMO_DIR, "temp", "discover_train_data.txt")
MODEL_CONFIG = os.path.join(DEMO_DIR, "model_discover_esmm.yaml")


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


def evaluate(model, dag, df, task_names: list[str], batch_size: int) -> dict:
    model.eval()
    all_logits: dict[str, list] = {t: [] for t in task_names}
    all_labels: dict[str, list] = {t: [] for t in task_names}
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.slice(start, batch_size)
            tensors = dag.preprocess_batch(batch_df.to_dicts())
            outputs = model(tensors)
            for t in task_names:
                if t in outputs:
                    all_logits[t].append(outputs[t].cpu().numpy().flatten())
                    label_col = t if t in batch_df.columns else "ctr"
                    if label_col in batch_df.columns:
                        all_labels[t].append(batch_df[label_col].to_numpy().astype(np.float32))
    metrics = {}
    for t in task_names:
        if all_logits[t] and all_labels[t]:
            logits = np.concatenate(all_logits[t])
            y = np.concatenate(all_labels[t])
            p = sigmoid(logits)
            # accuracy
            acc = float(((p > 0.5).astype(np.float32) == y).mean())
            metrics[t] = {"auc": auc_score(y, p), "accuracy": acc}
    return metrics


def train_epoch(model, optimizer, dag, df, task_names, batch_size, label_col_map) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(df), batch_size):
        batch_df = df.slice(start, batch_size)
        actual_bs = len(batch_df)
        tensors = dag.preprocess_batch(batch_df.to_dicts())
        outputs = model(tensors)
        loss = None
        for task_name, logits in outputs.items():
            label_col = label_col_map.get(task_name)
            if label_col is None or label_col not in batch_df.columns:
                continue
            labels = torch.tensor(batch_df[label_col].to_numpy(), dtype=torch.float32).view(
                actual_bs, 1
            )
            task_loss = F.binary_cross_entropy_with_logits(logits, labels)
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
    args = parser.parse_args()

    # ── Load data (Tab 分隔, 与真实数据格式一致) ──
    df = pl.read_csv(args.data, separator="\t").with_columns(
        [
            pl.col("ctr").cast(pl.Int64),
            pl.col("cvr").cast(pl.Int64),
        ]
    )
    df = df.sample(fraction=1.0, seed=42)
    n_train = int(len(df) * 0.8)
    train_df = df.slice(0, n_train)
    test_df = df.slice(n_train, len(df) - n_train)
    print(f"[Data] train={len(train_df)} test={len(test_df)}")

    # ── Build DAG ──
    flow_config = FlowConfig.from_yaml(args.feature_config)
    print(f"[Config] {len(flow_config.sources)} sources, {len(flow_config.operators)} operators")
    dag = FeatureDag(flow_config)
    features = dag.feature_tuples()
    print(f"[DAG] {len(features)} embeddable features")

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
        metrics = evaluate(model, dag, test_df, task_names, args.batch_size)

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

    # ── Inference demo: 用测试集前 5 条验证推理输 ──
    print("\n[Inference Demo] running on first 5 test samples (no labels)...")
    demo_df = test_df.slice(0, 5).drop(["ctr", "cvr", "is_click_detail", "is_click_stock"])
    model.eval()
    with torch.no_grad():
        tensors = dag.preprocess_batch(demo_df.to_dicts())
        preds = model(tensors)
        for task in task_names:
            if task in preds:
                logits = preds[task].cpu().numpy().flatten()
                probs = sigmoid(logits)
                print(f"  {task}: logits={logits.round(4)} probs={probs.round(4)}")

    print("Done.")


if __name__ == "__main__":
    main()
