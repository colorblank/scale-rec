from __future__ import annotations

"""统一训练入口：遍历所有模型，训练、评估、导出，保存预测供 Rust 验证。"""
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


def _wrap_unimixer(model):
    """Wrap UniMixer so state_dict matches Rust vb.pp("unimixer") prefix.

    Rust ModelConfig::build for UniMixer uses vb.pp("unimixer") for model internals
    but Python UniMixerModel stores blocks/task_towers directly. This wrapper nests
    them under an "unimixer" submodule to align state_dict keys.
    """
    import torch.nn as nn

    # Detach submodules from raw model before wrapping to avoid shared-tensor errors
    blocks = model.blocks
    task_towers = model.task_towers
    final_norm = model.final_norm
    tokenizer = model.tokenizer

    wrapper = nn.Module()
    wrapper.add_module("tokenizer", tokenizer)
    inner = nn.Module()
    inner.add_module("blocks", blocks)
    inner.add_module("task_towers", task_towers)
    if final_norm is not None:
        inner.add_module("final_norm", final_norm)
    wrapper.add_module("unimixer", inner)

    # Forward function uses the inner submodules directly
    def forward(self, x_inputs, temperature=None):
        t = temperature if temperature is not None else model.temperature
        tokens = self.tokenizer(x_inputs)
        bs = tokens.shape[0]
        x = tokens.reshape(bs, model.embed_dim)
        if model.use_siamese:
            x_bar = y_bar = x
            for blk in self.unimixer.blocks:
                _, xbn, ybn = blk(x, t, x_bar, y_bar, use_siamese=True)
                x_bar = xbn
                y_bar = ybn
                x = x_bar
            output = self.unimixer.final_norm(x_bar, y_bar, None)
        else:
            for blk in self.unimixer.blocks:
                x = blk(x, t, use_siamese=False)
            output = x
        return self.unimixer.task_towers(output)

    # Bind forward method
    import types

    wrapper.forward = types.MethodType(forward, wrapper)
    return wrapper


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_CONFIG = os.path.join(DEMO_DIR, "feature_config_demo.yaml")
DATA_PATH = os.path.join(DEMO_DIR, "temp", "train_data.csv")

MODELS = [
    ("lr", "model_lr_demo.yaml", ["ctr"]),
    ("deepfm", "model_deepfm_demo.yaml", ["ctr"]),
    ("mmoe", "model_mmoe_demo.yaml", ["ctr", "cvr"]),
    ("esmm", "model_esmm_demo.yaml", ["ctr", "cvr"]),
    ("unimixer", "model_unimixer_demo.yaml", ["ctr", "cvr"]),
]

SINGLE_TASK_LABEL = "ctr"


def evaluate(model, dag, df, task_names: list[str], batch_size: int) -> dict[str, dict]:
    """Evaluate model; returns {task_name: {logloss, auc, acc}}."""
    model.eval()
    all_outputs: dict[str, list] = {t: [] for t in task_names}
    all_labels: dict[str, list] = {t: [] for t in task_names}
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.slice(start, batch_size)
            feature_tensors = dag.preprocess_batch(batch_df.to_dicts())
            outputs = model(feature_tensors)
            for t in task_names:
                if t in outputs:
                    all_outputs[t].append(outputs[t].cpu().numpy().flatten())
                    labels_col = t if t in batch_df.columns else SINGLE_TASK_LABEL
                    if labels_col in batch_df.columns:
                        all_labels[t].append(batch_df[labels_col].to_numpy().astype(np.float32))
    results = {}
    for t in task_names:
        if all_outputs[t]:
            logits_arr = np.concatenate(all_outputs[t])
            labels_arr = (
                np.concatenate(all_labels[t]) if all_labels[t] else np.zeros_like(logits_arr)
            )
            probs = sigmoid(logits_arr)
            results[t] = {
                "logloss": logloss(labels_arr, probs),
                "auc": auc(labels_arr, probs),
                "accuracy": accuracy(labels_arr, probs),
            }
    return results


def predict_all(model, dag, df, batch_size: int) -> dict[str, np.ndarray]:
    """Return {output_key: logits_array} for all model outputs."""
    model.eval()
    all_keys = None
    all_logits: dict[str, list] = {}
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.slice(start, batch_size)
            feature_tensors = dag.preprocess_batch(batch_df.to_dicts())
            outputs = model(feature_tensors)
            if all_keys is None:
                all_keys = list(outputs.keys())
                all_logits = {k: [] for k in all_keys}
            for k in all_keys:
                all_logits[k].append(outputs[k].cpu().numpy().flatten())
    return {k: np.concatenate(v) for k, v in all_logits.items()}


def train_epoch(
    model, optimizer, dag, df, task_names: list[str], batch_size: int, label_col_map: dict
) -> float:
    """Train one epoch; returns average loss across all tasks."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(df), batch_size):
        batch_df = df.slice(start, batch_size)
        actual_bs = len(batch_df)
        feature_tensors = dag.preprocess_batch(batch_df.to_dicts())
        outputs = model(feature_tensors)
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


def train_one_model(model_type: str, model_config_path: str, label_cols: list[str], args) -> dict:
    """Train, evaluate, and export one model. Returns summary dict."""
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_type}")
    print(f"{'=' * 60}")

    # Load data
    df = pl.read_csv(args.data).with_columns(
        [
            pl.col("user_id").cast(pl.Int64),
            pl.col("ctr").cast(pl.Int64),
            pl.col("cvr").cast(pl.Int64),
        ]
    )
    df_shuffled = df.sample(fraction=1.0, seed=42)
    n_train = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.slice(0, n_train)
    test_df = df_shuffled.slice(n_train, len(df_shuffled) - n_train)
    print(f"[Data] train={len(train_df)} test={len(test_df)}")

    # Build DAG
    flow_config = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(flow_config)
    features = dag.feature_tuples()

    # Build model config
    model_config = ModelConfig.from_yaml(model_config_path)

    # Build tokenizer for UniMixer
    tokenizer = None
    if model_type == "unimixer":
        from train.models.unimixer.tokenizer import FeatureTokenizer

        tokenizer = FeatureTokenizer(features, model_config.token_dim, model_config.num_tokens)

    model = model_config.build(features, tokenizer=tokenizer)
    # Wrap UniMixer for Rust compatibility: Rust uses vb.pp("unimixer") prefix.
    # Python state_dict must have "unimixer.*" keys to match Candle VarBuilder paths.
    if model_type == "unimixer":
        model = _wrap_unimixer(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params={n_params:,}")

    # Determine task names for evaluation and label column mapping
    # Single-task models (LR, DeepFM): output "pred", label "ctr"
    # Multi-task models: output task names, label same name
    if model_type in ("lr", "deepfm"):
        eval_task_names = ["pred"]
        label_col_map = {"pred": "ctr"}
    elif model_type == "mmoe":
        eval_task_names = label_cols
        label_col_map = {t: t for t in label_cols}
    elif model_type == "esmm":
        eval_task_names = ["ctr", "cvr"]
        label_col_map = {"ctr": "ctr", "cvr": "cvr"}
    else:  # unimixer
        eval_task_names = ["ctr", "cvr"]
        label_col_map = {"ctr": "ctr", "cvr": "cvr"}

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, optimizer, dag, train_df, eval_task_names, args.batch_size, label_col_map
        )
        metrics = evaluate(model, dag, test_df, eval_task_names, args.batch_size)

        parts = [f"epoch {epoch:3d}/{args.epochs}  loss={train_loss:.6f}"]
        for t, m in metrics.items():
            if m["auc"] > best_auc:
                best_auc = m["auc"]
            parts.append(f"{t}: auc={m['auc']:.4f} acc={m['accuracy']:.4f}")
        print("  " + "  ".join(parts))

    print(f"[Best] AUC={best_auc:.4f}")

    # Export
    prefix = os.path.join(DEMO_DIR, "temp", f"model_{model_type}")
    safetensors_path = prefix + ".safetensors"
    test_csv_path = prefix + "_test.csv"
    preds_csv_path = prefix + "_py_preds.csv"

    export_to_safetensors(model, safetensors_path)
    test_df.write_csv(test_csv_path)

    all_preds = predict_all(model, dag, test_df, args.batch_size)
    preds_rows = {"label_ctr": test_df["ctr"].to_numpy().astype(np.float32)}
    if "cvr" in test_df.columns:
        preds_rows["label_cvr"] = test_df["cvr"].to_numpy().astype(np.float32)
    for k, v in all_preds.items():
        preds_rows[f"logit_{k}"] = v
    pl.DataFrame(preds_rows).write_csv(preds_csv_path)

    print(f"[Export] {safetensors_path}")
    print(f"[Export] {preds_csv_path}")

    return {"model_type": model_type, "best_auc": best_auc, "params": n_params}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-config", default=FEATURE_CONFIG)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--models", default="all", help="Comma-separated model types or 'all'")
    args = parser.parse_args()

    if args.models == "all":
        selected = MODELS
    else:
        wanted = set(args.models.split(","))
        selected = [(t, c, lbs) for t, c, lbs in MODELS if t in wanted]

    results = []
    for model_type, config_name, lbs in selected:
        config_path = os.path.join(DEMO_DIR, config_name)
        if not os.path.exists(config_path):
            print(f"[Skip] config not found: {config_path}")
            continue
        result = train_one_model(model_type, config_path, lbs, args)
        results.append(result)

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r["best_auc"] > 0.6 else "LOW"
        print(f"  {r['model_type']:<12} AUC={r['best_auc']:.4f}  params={r['params']:,}  {status}")
    print("Done.")


if __name__ == "__main__":
    main()
