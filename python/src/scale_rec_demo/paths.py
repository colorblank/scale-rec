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
SHARED_EXAMPLES_DIR = EXAMPLES_DIR / "shared"
MODEL_EXAMPLES_DIR = EXAMPLES_DIR / "models"
DISCOVER_FEATURE_CONFIG = SHARED_EXAMPLES_DIR / "feature_config_discover.yaml"

MODEL_CONFIGS = {
    "discover_lr": MODEL_EXAMPLES_DIR / "lr.yaml",
    "discover_gdcn_esmm": MODEL_EXAMPLES_DIR / "gdcn_esmm.yaml",
    "discover_unimixer": MODEL_EXAMPLES_DIR / "unimixer.yaml",
    "discover_token_mixer_large": MODEL_EXAMPLES_DIR / "token_mixer_large.yaml",
}
