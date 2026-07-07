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
DEMO_FEATURE_CONFIG = SHARED_EXAMPLES_DIR / "feature_config_demo.yaml"

MODEL_CONFIGS = {
    "demo_finalmlp": MODEL_EXAMPLES_DIR / "finalmlp.yaml",
    "demo_dcnv2": MODEL_EXAMPLES_DIR / "dcnv2.yaml",
    "demo_din": MODEL_EXAMPLES_DIR / "din.yaml",
    "demo_lr": MODEL_EXAMPLES_DIR / "lr.yaml",
    "demo_deepfm": MODEL_EXAMPLES_DIR / "deepfm.yaml",
    "demo_mmoe": MODEL_EXAMPLES_DIR / "mmoe.yaml",
    "demo_esmm": MODEL_EXAMPLES_DIR / "esmm_output_contract.yaml",
    "demo_gdcn_esmm": MODEL_EXAMPLES_DIR / "gdcn_esmm.yaml",
    "demo_unimixer": MODEL_EXAMPLES_DIR / "unimixer.yaml",
    "demo_token_mixer_large": MODEL_EXAMPLES_DIR / "token_mixer_large.yaml",
    "demo_rankmixer": MODEL_EXAMPLES_DIR / "rankmixer.yaml",
    "demo_full_mix": MODEL_EXAMPLES_DIR / "full_mix.yaml",
    "demo_rankup": MODEL_EXAMPLES_DIR / "rankup.yaml",
    "demo_hyformer": MODEL_EXAMPLES_DIR / "hyformer.yaml",
    "demo_fat": MODEL_EXAMPLES_DIR / "fat.yaml",
    "demo_mixformer": MODEL_EXAMPLES_DIR / "mixformer.yaml",
    "demo_onerank": MODEL_EXAMPLES_DIR / "onerank.yaml",
    "demo_onetrans": MODEL_EXAMPLES_DIR / "onetrans.yaml",
    "demo_uniformer": MODEL_EXAMPLES_DIR / "uniformer.yaml",
    "demo_pepnet": MODEL_EXAMPLES_DIR / "pepnet.yaml",
}
