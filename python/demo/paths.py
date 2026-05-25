from __future__ import annotations

"""Demo 配置路径索引。"""

from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent
TEMP_DIR = DEMO_DIR / "temp"
CONFIG_DIR = DEMO_DIR / "configs"
LEGACY_CONFIG_DIR = CONFIG_DIR / "legacy"
DISCOVER_CONFIG_DIR = CONFIG_DIR / "discover"

LEGACY_FEATURE_CONFIG = LEGACY_CONFIG_DIR / "feature_config.yaml"
DISCOVER_FEATURE_CONFIG = REPO_ROOT / "examples" / "feature_config_discover.yaml"

MODEL_CONFIGS = {
    "lr": LEGACY_CONFIG_DIR / "model_lr.yaml",
    "deepfm": LEGACY_CONFIG_DIR / "model_deepfm.yaml",
    "mmoe": LEGACY_CONFIG_DIR / "model_mmoe.yaml",
    "esmm": LEGACY_CONFIG_DIR / "model_esmm.yaml",
    "unimixer": LEGACY_CONFIG_DIR / "model_unimixer.yaml",
    "discover_esmm": DISCOVER_CONFIG_DIR / "model_esmm.yaml",
    "discover_unimixer": DISCOVER_CONFIG_DIR / "model_unimixer.yaml",
}
