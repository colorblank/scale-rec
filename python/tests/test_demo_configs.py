from __future__ import annotations

from pathlib import Path

import yaml

from train.core.config import FlowConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_demo_model_configs_exist_and_are_current():
    model_configs = {
        "discover_gdcn_esmm": EXAMPLES_DIR / "model_gdcn_esmm.yaml",
        "discover_unimixer": EXAMPLES_DIR / "model_discover_unimixer.yaml",
    }
    ancillary_configs = [
        EXAMPLES_DIR / "train_defaults.yaml",
        EXAMPLES_DIR / "discover_label_policy.yaml",
    ]
    expected_files = {path.name for path in model_configs.values()}
    actual_files = {path.name for path in EXAMPLES_DIR.glob("model*.yaml")}

    for path in list(model_configs.values()) + ancillary_configs:
        assert path.exists(), path
    assert actual_files == expected_files

    discover_fc = FlowConfig.from_yaml(str(EXAMPLES_DIR / "feature_config_discover.yaml"))
    assert [s.name for s in discover_fc.label_sources] == [
        "is_click",
        "is_cvr",
        "is_click_detail",
        "is_click_stock",
        "stay_time_label",
        "ctr",
        "cvr",
    ]
    assert len(discover_fc.feature_sources) == 38

    gdcn = yaml.safe_load(model_configs["discover_gdcn_esmm"].read_text(encoding="utf-8"))
    assert gdcn["type"] == "gdcn_esmm"
    assert [task["name"] for task in gdcn["tasks"]] == [
        "click",
        "cvr",
        "detail",
        "stock",
        "stay",
    ]
    assert gdcn["tasks"][-1]["metrics"] == ["mae", "mse"]
    task_config = gdcn["task_config"]
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
    assert gdcn["label_col_map"]["stay"] == "stay_time_label"

    unimixer = yaml.safe_load(model_configs["discover_unimixer"].read_text(encoding="utf-8"))
    assert unimixer["type"] == "unimixer"
    assert unimixer["use_siamese"] is False
    assert unimixer["tasks"][0]["metrics"] == ["auc", "logloss"]
    assert unimixer["label_col_map"]["stay"] == "stay_time_label"

    label_policy = yaml.safe_load(
        (EXAMPLES_DIR / "discover_label_policy.yaml").read_text(encoding="utf-8")
    )
    assert label_policy["click"]["threshold"] == 0.42
    assert label_policy["stay_time_label"]["noise_min"] == -25


def test_lr_ctr_duplicate_config_was_removed():
    assert not (REPO_ROOT / "python" / "demo" / "model_lr_ctr.yaml").exists()
