from __future__ import annotations

"""统一验证脚本：针对新增的 GDCN+ESMM 校验特征预处理和模型输出的一致性。"""

import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure python/src/ is in sys.path
_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.core.config import FlowConfig, ModelConfig
from train.core.dag import FeatureDag
from safetensors.torch import load_file

from .paths import DEMO_ARTIFACT_DIR, DISCOVER_FEATURE_CONFIG, MODEL_CONFIGS, REPO_ROOT

ALL_COLS = [
    "user_id",
    "item_id",
    "rec_algo",
    "is_click",
    "is_cvr",
    "is_click_detail",
    "is_click_stock",
    "stay_time",
    "p_date",
    "fav_securities",
    "recent_stocks",
    "interest_keywords",
    "follow_authors",
    "is_new_user",
    "hold_stocks",
    "hist_hold_stocks",
    "historical_click_items",
    "asset_level",
    "last_login_date",
    "city",
    "investment_horizon",
    "invest_style",
    "theme_interest",
    "industry_interest",
    "fund_favorites",
    "item_type",
    "roleneeds_first_label",
    "roleneeds_second_label",
    "invest_label",
    "invest_label_second",
    "invest_label_third",
    "quality_score_label",
    "stock_list",
    "entity_words_label",
    "item_entities_v3",
    "author_id",
    "author",
    "source_name",
    "emb_id",
    "wordnum",
    "answerscore",
    "has_picture",
    "has_video",
]


def add_header_to_tsv(input_path: Path, output_path: Path) -> None:
    """Prepend the columns header to the discover TSV dataset."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(ALL_COLS) + "\n")
        f.write(content)
    print(f"[Prep] Prepended header to TSV dataset: {output_path}")


def run_training(
    data_path: Path, feature_config: Path, model_config: Path, weights_path: Path
) -> None:
    """Train GDCN+ESMM model for 1 epoch to save safetensors weights."""
    print("[Train] Training discover gdcn_esmm for 1 epoch...")
    cmd = [
        "python",
        "-m",
        "train.main",
        "discover",
        "--data",
        str(data_path),
        "--feature-config",
        str(feature_config),
        "--model-config",
        str(model_config),
        "--epochs",
        "1",
        "--batch-size",
        "128",
        "--lr",
        "0.005",
        "--export-path",
        str(weights_path),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def generate_pytorch_predictions(
    feature_config: Path,
    model_config: Path,
    weights_path: Path,
    data_path: Path,
    test_csv: Path,
    py_preds_csv: Path,
) -> None:
    """Run PyTorch inference on the first 100 rows and save predictions."""
    print("[PyTorch] Generating PyTorch prediction outputs...")
    # Load dataset
    df = pd.read_csv(data_path, sep="\t").head(100)

    # Build models
    fc = FlowConfig.from_yaml(str(feature_config))
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    m_cfg = ModelConfig.from_yaml(model_config)
    model = m_cfg.build(features)

    # Enforce all defined sources exist and replace NaNs with configured default values
    for source in fc.sources:
        if source.name in df.columns:
            df[source.name] = df[source.name].fillna(source.default_val)
        else:
            df[source.name] = source.default_val

    # Save the test inputs with all configured source columns
    df.to_csv(test_csv, index=False)

    # Load weights
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    model.eval()

    # Inference
    with torch.no_grad():
        feature_tensors = dag.preprocess_batch(df.to_dict("records"))
        outputs = model(feature_tensors)

    # Write PyTorch prediction CSV
    preds_rows = {}
    for key, logits in outputs.items():
        preds_rows[f"logit_{key}"] = logits.cpu().numpy().flatten()
    pd.DataFrame(preds_rows).to_csv(py_preds_csv, index=False)
    print(f"[PyTorch] Saved predictions to {py_preds_csv}")


def run_rust_inference(
    feature_config: Path,
    model_config: Path,
    weights_path: Path,
    test_csv: Path,
    rust_preds_csv: Path,
) -> None:
    """Invoke Rust demo_inference bin to generate Rust predictions."""
    print("[Rust] Invoking Rust inference engine...")
    cmd = [
        "cargo",
        "run",
        "--bin",
        "demo_inference",
        "--",
        str(feature_config),
        str(model_config),
        str(weights_path),
        str(test_csv),
        str(rust_preds_csv),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def compare_outputs(py_preds_csv: Path, rust_preds_csv: Path, threshold: float = 1e-4) -> bool:
    """Compare PyTorch vs Rust prediction logits."""
    print("\n" + "=" * 50)
    print("  PyTorch vs Rust Outputs Comparison")
    print("=" * 50)
    py_df = pd.read_csv(py_preds_csv)
    rust_df = pd.read_csv(rust_preds_csv)

    all_pass = True
    for col in py_df.columns:
        if col not in rust_df.columns:
            print(f"  FAIL: Column '{col}' is missing in Rust output!")
            all_pass = False
            continue

        py_arr = py_df[col].to_numpy()
        rust_arr = rust_df[col].to_numpy()

        diff = np.abs(py_arr - rust_arr)
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        status = "PASS" if max_diff < threshold else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {col:<18} max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  {status}")

    print("-" * 50)
    print(f"  Overall Consistency Check: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main() -> None:
    raw_data_path = DEMO_ARTIFACT_DIR / "discover_train_data.txt"
    if not raw_data_path.exists():
        print(f"[Error] missing raw discover dataset: {raw_data_path}")
        sys.exit(1)

    data_header_path = DEMO_ARTIFACT_DIR / "discover_train_data_header.txt"
    add_header_to_tsv(raw_data_path, data_header_path)

    feature_cfg_path = DISCOVER_FEATURE_CONFIG
    model_cfg_path = MODEL_CONFIGS["discover_gdcn_esmm"]
    weights_path = DEMO_ARTIFACT_DIR / "gdcn_esmm.safetensors"

    # 1. Run 1-epoch training
    run_training(data_header_path, feature_cfg_path, model_cfg_path, weights_path)

    test_csv = DEMO_ARTIFACT_DIR / "gdcn_esmm_test.csv"
    py_preds_csv = DEMO_ARTIFACT_DIR / "gdcn_esmm_py_preds.csv"
    rust_preds_csv = DEMO_ARTIFACT_DIR / "gdcn_esmm_rust_preds.csv"

    # 2. PyTorch predictions
    generate_pytorch_predictions(
        feature_cfg_path, model_cfg_path, weights_path, data_header_path, test_csv, py_preds_csv
    )

    # 3. Rust predictions
    run_rust_inference(feature_cfg_path, model_cfg_path, weights_path, test_csv, rust_preds_csv)

    # 4. Compare outputs
    all_pass = compare_outputs(py_preds_csv, rust_preds_csv)

    # Clean up temporary headers if desired, keeping outputs for verification
    if data_header_path.exists():
        data_header_path.unlink()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
