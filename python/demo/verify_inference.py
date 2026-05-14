from __future__ import annotations

"""PyTorch vs Rust 推理一致性验证：比较 logits 并输出报告。"""
import os
import subprocess
import sys

import numpy as np
import polars as pl


def compare_logits(py_logits: np.ndarray, rust_logits: np.ndarray) -> dict:
    """Compare two logit vectors and return metrics dict."""
    diff = np.abs(py_logits - rust_logits)
    corr = float(np.corrcoef(py_logits, rust_logits)[0, 1])
    sign_mismatches = int(np.sum(np.sign(py_logits) != np.sign(rust_logits)))
    return {
        "samples": len(py_logits),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "correlation": corr,
        "sign_mismatches": sign_mismatches,
    }


def main() -> None:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    demo_dir = os.path.abspath(os.path.dirname(__file__))
    temp_dir = os.path.join(demo_dir, "temp")
    feature_config = os.path.join(demo_dir, "feature_config_demo.yaml")
    model_config = os.path.join(demo_dir, "model_lr_demo.yaml")
    safetensors = os.path.join(temp_dir, "model_lr.safetensors")
    test_csv = os.path.join(temp_dir, "model_lr_test.csv")
    py_preds = os.path.join(temp_dir, "model_lr_py_preds.csv")
    rust_preds = os.path.join(temp_dir, "model_lr_rust_preds.csv")

    # Run Rust inference if not done
    if not os.path.exists(rust_preds):
        print("[Verify] Running Rust inference...")
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--bin",
                "demo_inference",
                "--",
                feature_config,
                model_config,
                safetensors,
                test_csv,
                rust_preds,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[Verify] Rust inference failed:\n{result.stderr}")
            sys.exit(1)
        print(result.stdout)

    # Load predictions
    py_df = pl.read_csv(py_preds)
    rust_df = pl.read_csv(rust_preds)

    if len(py_df) != len(rust_df):
        print(
            f"[Verify] ERROR: row count mismatch: PyTorch={len(py_df)}, Rust={len(rust_df)}"
        )
        sys.exit(1)

    py_logits = py_df["logit"].to_numpy()
    rust_logits = rust_df["logit"].to_numpy()
    labels = py_df["label"].to_numpy()

    # Compute metrics
    metrics = compare_logits(py_logits, rust_logits)

    # Also compute per-sample stats
    probs_py = 1.0 / (1.0 + np.exp(-py_logits))
    probs_rust = 1.0 / (1.0 + np.exp(-rust_logits))
    prob_diff = np.abs(probs_py - probs_rust)

    # Report
    print("=" * 60)
    print("  PyTorch vs Rust 推理一致性验证报告")
    print("=" * 60)
    print(f"  样本数:          {metrics['samples']}")
    print(f"  PyTorch logit:   mean={py_logits.mean():.4f}  std={py_logits.std():.4f}")
    print(f"  Rust   logit:    mean={rust_logits.mean():.4f}  std={rust_logits.std():.4f}")
    print(f"  最大绝对差:      {metrics['max_abs_diff']:.2e}")
    print(f"  平均绝对差:      {metrics['mean_abs_diff']:.2e}")
    print(f"  Pearson 相关系数: {metrics['correlation']:.8f}")
    print(f"  符号不一致数:     {metrics['sign_mismatches']} / {metrics['samples']}")
    print(f"  概率最大差:       {prob_diff.max():.2e}")
    print(f"  概率平均差:       {prob_diff.mean():.2e}")
    print("-" * 60)

    threshold = 1e-3
    if metrics["max_abs_diff"] < threshold:
        print(f"  结果: PASS  (max_abs_diff={metrics['max_abs_diff']:.2e} < {threshold})")
    else:
        print(
            f"  结果: FAIL  (max_abs_diff={metrics['max_abs_diff']:.2e} >= {threshold})"
        )
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
