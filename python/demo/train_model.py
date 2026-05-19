from __future__ import annotations

"""DeepFM 训练与评估：加载合成数据，训练，评估，导出权重。"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

# 确保 src/ 在 sys.path 中，使 `train` 包可导入（不依赖 .pth 或 venv 配置）
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.export import export_to_safetensors  # noqa: E402
from train.models import ModelConfig  # noqa: E402

from ._metrics import accuracy, auc, logloss, sigmoid  # noqa: E402

LABEL_COL = "ctr"


def evaluate(model, dag, df: pl.DataFrame, batch_size: int) -> dict[str, float]:
    """Evaluate model on DataFrame; return logloss, auc, acc."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.slice(start, batch_size)
            actual_bs = len(batch_df)
            feature_tensors = dag.preprocess_batch(batch_df.to_dicts())
            outputs = model(feature_tensors)
            logits = outputs["pred"].cpu().numpy().flatten()
            labels = batch_df[LABEL_COL].to_numpy().astype(np.float32)
            if len(labels) < actual_bs:
                labels = np.pad(labels, (0, actual_bs - len(labels)), constant_values=0)
            all_logits.append(logits)
            all_labels.append(labels)
    logits_arr = np.concatenate(all_logits)
    labels_arr = np.concatenate(all_labels)
    probs = sigmoid(logits_arr)
    return {
        "logloss": logloss(labels_arr, probs),
        "auc": auc(labels_arr, probs),
        "accuracy": accuracy(labels_arr, probs),
    }


def predict_logits(model, dag, df: pl.DataFrame, batch_size: int) -> np.ndarray:
    """Return raw logits for all rows."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.slice(start, batch_size)
            feature_tensors = dag.preprocess_batch(batch_df.to_dicts())
            outputs = model(feature_tensors)
            all_logits.append(outputs["pred"].cpu().numpy().flatten())
    return np.concatenate(all_logits)


def train_epoch(model, optimizer, dag, df: pl.DataFrame, batch_size: int) -> float:
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(df), batch_size):
        batch_df = df.slice(start, batch_size)
        actual_bs = len(batch_df)
        feature_tensors = dag.preprocess_batch(batch_df.to_dicts())
        outputs = model(feature_tensors)
        labels = torch.tensor(batch_df[LABEL_COL].to_numpy(), dtype=torch.float32).view(
            actual_bs, 1
        )
        loss = F.binary_cross_entropy_with_logits(outputs["pred"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    demo_dir = os.path.dirname(__file__) or "."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-config",
        default=os.path.join(demo_dir, "feature_config_demo.yaml"),
    )
    temp_dir = os.path.join(demo_dir, "temp")
    parser.add_argument(
        "--model-config",
        default=os.path.join(demo_dir, "model_lr_demo.yaml"),
    )
    parser.add_argument("--data", default=os.path.join(temp_dir, "train_data.csv"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--export-path", default=os.path.join(temp_dir, "model_lr.safetensors"))
    parser.add_argument("--test-data-out", default=os.path.join(temp_dir, "model_lr_test.csv"))
    parser.add_argument("--preds-out", default=os.path.join(temp_dir, "model_lr_py_preds.csv"))
    args = parser.parse_args()

    df = pl.read_csv(args.data)
    print(f"[Data] {len(df)} rows, columns={df.columns}")

    # Ensure correct dtypes for numeric columns
    df = df.with_columns(
        [
            pl.col("user_id").cast(pl.Int64),
            pl.col(LABEL_COL).cast(pl.Int64),
        ]
    )

    # Train/test split 80/20 (shuffled to mix user IDs)
    df_shuffled = df.sample(fraction=1.0, seed=42)
    n_train = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.slice(0, n_train)
    test_df = df_shuffled.slice(n_train, len(df_shuffled) - n_train)
    print(f"[Split] train={len(train_df)}, test={len(test_df)}")

    flow_config = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(flow_config)
    features = dag.feature_tuples()
    print(f"[DAG] {len(features)} embeddable features:")
    for name, vocab, dim in features:
        print(f"  {name:<25} vocab={vocab:<8} dim={dim}")

    model_config = ModelConfig.from_yaml(args.model_config)
    model = model_config.build(features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] DeepFM params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, optimizer, dag, train_df, args.batch_size)
        metrics = evaluate(model, dag, test_df, args.batch_size)
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
        print(
            f"  epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_loss:.6f}  "
            f"logloss={metrics['logloss']:.6f}  "
            f"auc={metrics['auc']:.4f}  "
            f"acc={metrics['accuracy']:.4f}"
        )

    print(f"[Best] AUC={best_auc:.4f}")

    # Export safetensors
    export_to_safetensors(model, args.export_path)

    # Save test data and PyTorch predictions for verification
    test_df.write_csv(args.test_data_out)
    logits = predict_logits(model, dag, test_df, args.batch_size)
    labels_arr = test_df[LABEL_COL].to_numpy().astype(np.float32)
    preds_df = pl.DataFrame(
        {
            "label": labels_arr,
            "logit": logits.tolist(),
        }
    )
    preds_df.write_csv(args.preds_out)
    print(f"[Export] test_data → {args.test_data_out}")
    print(f"[Export] predictions → {args.preds_out}")
    print("Done.")


if __name__ == "__main__":
    main()
