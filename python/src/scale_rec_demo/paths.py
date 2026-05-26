from __future__ import annotations

"""统一示例配置路径索引。"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PYTHON_SRC_DIR = PACKAGE_DIR.parent
PYTHON_DIR = PYTHON_SRC_DIR.parent
REPO_ROOT = PYTHON_DIR.parent

ARTIFACT_DIR = PYTHON_DIR / "artifacts"
DEMO_ARTIFACT_DIR = ARTIFACT_DIR / "demo"
TEMP_DIR = DEMO_ARTIFACT_DIR

EXAMPLES_DIR = REPO_ROOT / "examples"
LEGACY_FEATURE_CONFIG = EXAMPLES_DIR / "feature_config_legacy.yaml"
DISCOVER_FEATURE_CONFIG = EXAMPLES_DIR / "feature_config_discover.yaml"

MODEL_CONFIGS = {
    "lr": EXAMPLES_DIR / "model_lr.yaml",
    "deepfm": EXAMPLES_DIR / "model_deepfm.yaml",
    "mmoe": EXAMPLES_DIR / "model_mmoe.yaml",
    "esmm": EXAMPLES_DIR / "model_esmm.yaml",
    "unimixer": EXAMPLES_DIR / "model_unimixer.yaml",
    "discover_esmm": EXAMPLES_DIR / "model_discover_esmm.yaml",
    "discover_gdcn_esmm": EXAMPLES_DIR / "model_gdcn_esmm.yaml",
    "discover_unimixer": EXAMPLES_DIR / "model_discover_unimixer.yaml",
}
