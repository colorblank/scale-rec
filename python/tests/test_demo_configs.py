from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_demo_model_configs_exist_and_are_current():
    configs = {
        "lr": EXAMPLES_DIR / "model_lr.yaml",
        "deepfm": EXAMPLES_DIR / "model_deepfm.yaml",
        "mmoe": EXAMPLES_DIR / "model_mmoe.yaml",
        "esmm": EXAMPLES_DIR / "model_esmm.yaml",
        "unimixer": EXAMPLES_DIR / "model_unimixer.yaml",
        "discover_gdcn_esmm": EXAMPLES_DIR / "model_gdcn_esmm.yaml",
        "discover_unimixer": EXAMPLES_DIR / "model_discover_unimixer.yaml",
    }

    for path in configs.values():
        assert path.exists(), path

    esmm = yaml.safe_load(configs["esmm"].read_text(encoding="utf-8"))
    assert "ctr_hidden_dims" not in esmm
    task_config = esmm["task_config"]
    assert [tower["name"] for tower in task_config["towers"]] == [
        "click",
        "cvr",
        "detail",
        "stock",
        "stay",
    ]
    assert {relation["target"] for relation in task_config["relations"]} == {
        "ctcvr",
        "ctdetail",
        "ctstock",
        "ctstay",
    }


def test_lr_ctr_duplicate_config_was_removed():
    assert not (REPO_ROOT / "python" / "demo" / "model_lr_ctr.yaml").exists()
