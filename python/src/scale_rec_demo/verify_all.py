from __future__ import annotations

"""Unified verification script comparing PyTorch vs Rust inference outputs."""

import contextlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

# Ensure python/src/ is in sys.path
_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.app.cli import build_model_for_dag
from train.app.data import DTYPE_PANDAS, _parse_default
from train.app.main import _run_discover as run_discover_in_process
from train.app.main import build_parser as build_train_parser
from train.core.config import FlowConfig
from train.core.dag import FeatureDag

from .paths import DEMO_ARTIFACT_DIR, DISCOVER_FEATURE_CONFIG, MODEL_CONFIGS, REPO_ROOT


def add_header_to_tsv(input_path: Path, output_path: Path, columns: list[str]) -> None:
    """Prepend the columns header to a TSV dataset."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(columns) + "\n" + content)


def read_configured_dataframe(data_path: Path, flow_config: FlowConfig) -> pd.DataFrame:
    """Read TSV using the same source dtypes/defaults as training."""
    dtype = {s.name: DTYPE_PANDAS.get(s.dtype.tag, "str") for s in flow_config.sources}
    defaults = {s.name: _parse_default(s.default_val, s.dtype.tag) for s in flow_config.sources}
    df = pd.read_csv(
        data_path,
        sep="\t",
        header=0,
        dtype=dtype,
        na_values=["NULL", "\\N", "null", "None", ""],
        keep_default_na=False,
    )
    for name, default_val in defaults.items():
        df[name] = df[name].fillna(default_val) if name in df.columns else default_val
    return df


def run_training_process(model_type: str, data_path: Path, weights_path: Path) -> None:
    """Train the model for 1 epoch to save safetensors weights."""
    print(f"[Train] Training {model_type} for 1 epoch...")
    stem = model_type.replace("discover_", "")
    args = build_train_parser().parse_args(
        [
            "discover",
            "--data",
            str(data_path),
            "--feature-config",
            str(DISCOVER_FEATURE_CONFIG),
            "--model-config",
            str(MODEL_CONFIGS[model_type]),
            "--epochs",
            "1",
            "--batch-size",
            "128",
            "--lr",
            "0.005",
            "--artifact-dir",
            str(DEMO_ARTIFACT_DIR),
            "--publish-path",
            str(weights_path),
            "--model-name",
            f"model_{stem}",
        ]
    )
    args.repo_root = REPO_ROOT
    run_discover_in_process(args)


def _load_pytorch_model(
    model_config: Path, fc: FlowConfig, weights_path: Path
) -> tuple[FeatureDag, torch.nn.Module]:
    """Load FeatureDag and PyTorch model with loaded weights."""
    dag = FeatureDag(fc)
    model = build_model_for_dag(model_config, dag, torch.device("cpu")).model
    model.load_state_dict(load_file(str(weights_path)))
    model.eval()
    return dag, model


def save_test_and_pytorch_preds(
    model_config: Path,
    fc: FlowConfig,
    weights_path: Path,
    data_path: Path,
    test_csv: Path,
    py_preds_csv: Path,
) -> None:
    """Generate PyTorch predictions and save test inputs and outputs."""
    dag, model = _load_pytorch_model(model_config, fc, weights_path)
    df = read_configured_dataframe(data_path, fc).head(100)
    df.to_csv(test_csv, index=False)
    with torch.no_grad():
        features = dag.preprocess_batch(df.to_dict("records"))
        outputs = model(features)
    preds = {f"logit_{k}": v.cpu().numpy().flatten() for k, v in outputs.items()}
    pd.DataFrame(preds).to_csv(py_preds_csv, index=False)


def run_rust_inference(
    model_type: str, weights_path: Path, test_csv: Path, rust_preds_csv: Path
) -> None:
    """Invoke Rust demo_inference bin to generate Rust predictions."""
    print(f"[Rust] Invoking Rust inference engine for {model_type}...")
    cmd = [
        "cargo",
        "run",
        "--bin",
        "demo_inference",
        "--",
        str(DISCOVER_FEATURE_CONFIG),
        str(MODEL_CONFIGS[model_type]),
        str(weights_path),
        str(test_csv),
        str(rust_preds_csv),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def compare_arrays(py_arr: np.ndarray, rust_arr: np.ndarray) -> dict:
    """Calculate correlation, max absolute difference, mean absolute difference, etc."""
    diff = np.abs(py_arr - rust_arr)
    corr = float(np.corrcoef(py_arr, rust_arr)[0, 1]) if len(py_arr) > 1 else 1.0
    sign_count = int(np.sum(np.sign(py_arr) != np.sign(rust_arr)))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "correlation": corr,
        "sign_mismatches": sign_count,
    }


def compare_outputs(
    py_preds_csv: Path, rust_preds_csv: Path, threshold: float
) -> tuple[bool, dict]:
    """Compare PyTorch vs Rust prediction logits and return validation results."""
    py_df, rust_df = pd.read_csv(py_preds_csv), pd.read_csv(rust_preds_csv)
    py_cols = [c for c in py_df.columns if c.startswith("logit_")]
    all_pass, metrics_summary = True, {}
    for col in py_cols:
        if col not in rust_df.columns:
            print(f"  FAIL: Column '{col}' is missing in Rust output!")
            all_pass = False
            continue
        m = compare_arrays(py_df[col].to_numpy(), rust_df[col].to_numpy())
        metrics_summary[col.replace("logit_", "")] = m
        status = "PASS" if m["max_abs_diff"] < threshold else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"  {col:<18} max_diff={m['max_abs_diff']:.2e} mean_diff={m['mean_abs_diff']:.2e} {status}"
        )
    return all_pass, metrics_summary


@contextlib.contextmanager
def temp_header_data(fc: FlowConfig):
    """Context manager to prepare headered training data and ensure cleanup."""
    raw_data = DEMO_ARTIFACT_DIR / "discover_train_data.txt"
    header_data = DEMO_ARTIFACT_DIR / "discover_train_data_header.txt"
    cols = [s.name for s in fc.sources]
    add_header_to_tsv(raw_data, header_data, cols)
    try:
        yield header_data
    finally:
        if header_data.exists():
            header_data.unlink()


def verify_model_pipeline(
    model_type: str, force_train: bool = False, threshold: float = 1e-4
) -> bool:
    """Run verification pipeline for a given model type."""
    print(f"\n{'=' * 60}\n  Model: {model_type}\n{'=' * 60}")
    stem = f"model_{model_type.removeprefix('discover_')}"
    weights = DEMO_ARTIFACT_DIR / f"{stem}.safetensors"
    test_csv = DEMO_ARTIFACT_DIR / f"{stem}_test.csv"
    py_preds = DEMO_ARTIFACT_DIR / f"{stem}_py_preds.csv"
    rust_preds = DEMO_ARTIFACT_DIR / f"{stem}_rust_preds.csv"
    fc = FlowConfig.from_yaml(str(DISCOVER_FEATURE_CONFIG))
    with temp_header_data(fc) as header_data:
        if force_train or not weights.exists():
            run_training_process(model_type, header_data, weights)
        save_test_and_pytorch_preds(
            MODEL_CONFIGS[model_type], fc, weights, header_data, test_csv, py_preds
        )
    run_rust_inference(model_type, weights, test_csv, rust_preds)
    success, _ = compare_outputs(py_preds, rust_preds, threshold)
    return success


def main() -> None:
    """Parse command line arguments and execute verification for all requested models."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify Python vs Rust prediction consistency.")
    parser.add_argument(
        "--models",
        default="discover_lr,discover_gdcn_esmm,discover_unimixer",
        help="Comma-separated models.",
    )
    parser.add_argument("--force-train", action="store_true", help="Force train models.")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Similarity threshold.")
    args = parser.parse_args()

    models = args.models.split(",")
    success = True
    for model in models:
        if not verify_model_pipeline(model.strip(), args.force_train, args.threshold):
            success = False

    print(f"\nOverall Consistency Status: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
