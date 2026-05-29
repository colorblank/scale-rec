from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_demo_model_configs_exist_and_are_current():
    configs = {
        "discover_gdcn_esmm": EXAMPLES_DIR / "model_gdcn_esmm.yaml",
        "discover_unimixer": EXAMPLES_DIR / "model_discover_unimixer.yaml",
    }
    expected_files = {path.name for path in configs.values()}
    actual_files = {path.name for path in EXAMPLES_DIR.glob("model*.yaml")}

    for path in configs.values():
        assert path.exists(), path
    assert actual_files == expected_files

    gdcn = yaml.safe_load(configs["discover_gdcn_esmm"].read_text(encoding="utf-8"))
    assert gdcn["type"] == "gdcn_esmm"
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

    unimixer = yaml.safe_load(configs["discover_unimixer"].read_text(encoding="utf-8"))
    assert unimixer["type"] == "unimixer"
    assert unimixer["use_siamese"] is False


def test_lr_ctr_duplicate_config_was_removed():
    assert not (REPO_ROOT / "python" / "demo" / "model_lr_ctr.yaml").exists()
