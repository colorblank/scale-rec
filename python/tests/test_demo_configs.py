from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "python" / "demo"
LEGACY_CONFIG_DIR = DEMO_DIR / "configs" / "legacy"
DISCOVER_CONFIG_DIR = DEMO_DIR / "configs" / "discover"


def test_demo_model_configs_exist_and_are_current():
    configs = {
        "lr": LEGACY_CONFIG_DIR / "model_lr.yaml",
        "deepfm": LEGACY_CONFIG_DIR / "model_deepfm.yaml",
        "mmoe": LEGACY_CONFIG_DIR / "model_mmoe.yaml",
        "esmm": LEGACY_CONFIG_DIR / "model_esmm.yaml",
        "unimixer": LEGACY_CONFIG_DIR / "model_unimixer.yaml",
        "discover_esmm": DISCOVER_CONFIG_DIR / "model_esmm.yaml",
        "discover_unimixer": DISCOVER_CONFIG_DIR / "model_unimixer.yaml",
    }

    for path in configs.values():
        assert path.exists(), path

    esmm = yaml.safe_load(configs["esmm"].read_text(encoding="utf-8"))
    assert "ctr_hidden_dims" not in esmm
    for key in [
        "click_hidden_dims",
        "cvr_hidden_dims",
        "detail_hidden_dims",
        "stock_hidden_dims",
        "stay_hidden_dims",
    ]:
        assert key in esmm


def test_lr_ctr_duplicate_config_was_removed():
    assert not (DEMO_DIR / "model_lr_ctr.yaml").exists()
