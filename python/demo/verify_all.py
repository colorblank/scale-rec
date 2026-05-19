from __future__ import annotations

"""统一验证脚本：对每个模型比较 PyTorch vs Rust 推理结果。"""
import os
import subprocess
import sys

import numpy as np
import polars as pl
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
FEATURE_CONFIG = os.path.join(DEMO_DIR, "feature_config_demo.yaml")

MODEL_TYPES = ["lr", "deepfm", "mmoe", "esmm", "unimixer"]


def _make_inference_config(base_path: str) -> str:
    """Create a temp feature config with StringConcatHash operators in inference mode."""
    with open(base_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    modified = False
    for op in config.get("operators", []):
        if op.get("op_type") == "StringConcatHash":
            params = op.setdefault("params", {})
            if params.get("mode", "train") == "train":
                params["mode"] = "inference"
                modified = True
    if not modified:
        return base_path
    out_path = os.path.join(DEMO_DIR, "temp", "_feature_config_infer.yaml")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    return out_path


def compare_arrays(py_arr: np.ndarray, rust_arr: np.ndarray) -> dict:
    diff = np.abs(py_arr - rust_arr)
    corr = float(np.corrcoef(py_arr, rust_arr)[0, 1])
    sign_count = int(np.sum(np.sign(py_arr) != np.sign(rust_arr)))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "correlation": corr,
        "sign_mismatches": sign_count,
    }


def verify_model(model_type: str, threshold: float = 1e-3) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_type}")
    print(f"{'=' * 60}")

    temp_dir = os.path.join(DEMO_DIR, "temp")
    model_config = os.path.join(DEMO_DIR, f"model_{model_type}_demo.yaml")
    safetensors = os.path.join(temp_dir, f"model_{model_type}.safetensors")
    test_csv = os.path.join(temp_dir, f"model_{model_type}_test.csv")
    py_preds_csv = os.path.join(temp_dir, f"model_{model_type}_py_preds.csv")
    rust_preds_csv = os.path.join(temp_dir, f"model_{model_type}_rust_preds.csv")

    # Check prerequisites
    for fpath, desc in [
        (safetensors, "safetensors"),
        (test_csv, "test csv"),
        (py_preds_csv, "python preds"),
    ]:
        if not os.path.exists(fpath):
            print(f"  SKIP: missing {desc}: {fpath}")
            return {"model_type": model_type, "status": "SKIP", "reason": f"missing {desc}"}

    # Run Rust inference（StringConcatHash 切换到 inference 模式）
    infer_config = _make_inference_config(FEATURE_CONFIG)
    print("  Running Rust inference...")
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--bin",
            "demo_inference",
            "--",
            infer_config,
            model_config,
            safetensors,
            test_csv,
            rust_preds_csv,
        ],
        cwd=PYTHON_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  FAIL: Rust inference error:\n{result.stderr}")
        return {"model_type": model_type, "status": "FAIL", "reason": result.stderr[:200]}

    # Load predictions
    py_df = pl.read_csv(py_preds_csv)
    rust_df = pl.read_csv(rust_preds_csv)

    # Find logit columns
    py_logit_cols = [c for c in py_df.columns if c.startswith("logit_")]
    rust_logit_cols = [c for c in rust_df.columns if c.startswith("logit_")]

    if len(py_logit_cols) != len(rust_logit_cols):
        print(f"  FAIL: column count mismatch: py={py_logit_cols} rust={rust_logit_cols}")
        return {"model_type": model_type, "status": "FAIL", "reason": "column count mismatch"}

    # Compare each output key
    all_pass = True
    key_results = {}
    for py_col in sorted(py_logit_cols):
        key = py_col.replace("logit_", "")
        rust_col = f"logit_{key}"
        if rust_col not in rust_logit_cols:
            print(f"  FAIL: missing rust column {rust_col}")
            all_pass = False
            continue

        py_arr = py_df[py_col].to_numpy()
        rust_arr = rust_df[rust_col].to_numpy()
        metrics = compare_arrays(py_arr, rust_arr)
        key_results[key] = metrics

        status = "PASS" if metrics["max_abs_diff"] < threshold else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  {key:<8} max_diff={metrics['max_abs_diff']:.2e}  "
            f"mean_diff={metrics['mean_abs_diff']:.2e}  "
            f"corr={metrics['correlation']:.8f}  "
            f"sign_mismatch={metrics['sign_mismatches']}  {status}"
        )

    overall = "PASS" if all_pass else "FAIL"
    print(f"  Overall: {overall}")
    return {"model_type": model_type, "status": overall, "keys": key_results}


def main() -> None:
    # Parse optional args
    models = MODEL_TYPES
    if len(sys.argv) > 1:
        models = sys.argv[1].split(",")

    results = []
    for mt in models:
        results.append(verify_model(mt))

    # Summary
    print(f"\n{'=' * 60}")
    print("  Verification Summary")
    print(f"{'=' * 60}")
    all_pass = True
    for r in results:
        status = r["status"]
        if status != "PASS":
            all_pass = False
        details = ""
        if "keys" in r:
            keys_info = ", ".join(f"{k}: max={v['max_abs_diff']:.2e}" for k, v in r["keys"].items())
            details = f"  [{keys_info}]"
        elif "reason" in r:
            details = f"  ({r['reason']})"
        print(f"  {r['model_type']:<12} {status:<6}{details}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  Overall: {overall}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
