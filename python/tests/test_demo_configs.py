from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scale_rec_demo.paths import MODEL_CONFIGS
from scale_rec_demo.verify_all import (
    compare_outputs,
    read_serving_dataframe,
    selected_model_names,
    serving_array,
)
from train.core.config import FlowConfig
from train.core.model_output import OutputTensor

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_demo_model_configs_exist_and_are_current():
    model_configs = {
        "demo_lr": EXAMPLES_DIR / "models" / "lr.yaml",
        "demo_deepfm": EXAMPLES_DIR / "models" / "deepfm.yaml",
        "demo_mmoe": EXAMPLES_DIR / "models" / "mmoe.yaml",
        "demo_esmm": EXAMPLES_DIR / "models" / "esmm_output_contract.yaml",
        "demo_gdcn_esmm": EXAMPLES_DIR / "models" / "gdcn_esmm.yaml",
        "demo_unimixer": EXAMPLES_DIR / "models" / "unimixer.yaml",
        "demo_token_mixer_large": EXAMPLES_DIR / "models" / "token_mixer_large.yaml",
        "demo_rankmixer": EXAMPLES_DIR / "models" / "rankmixer.yaml",
        "demo_rankup": EXAMPLES_DIR / "models" / "rankup.yaml",
        "demo_hyformer": EXAMPLES_DIR / "models" / "hyformer.yaml",
        "demo_fat": EXAMPLES_DIR / "models" / "fat.yaml",
        "demo_mixformer": EXAMPLES_DIR / "models" / "mixformer.yaml",
        "demo_onerank": EXAMPLES_DIR / "models" / "onerank.yaml",
        "demo_onetrans": EXAMPLES_DIR / "models" / "onetrans.yaml",
        "demo_pepnet": EXAMPLES_DIR / "models" / "pepnet.yaml",
    }
    ancillary_configs = [
        EXAMPLES_DIR / "shared" / "train_defaults.yaml",
        EXAMPLES_DIR / "shared" / "demo_label_policy.yaml",
        EXAMPLES_DIR / "shared" / "feature_config_demo.yaml",
    ]
    actual_files = {
        path.relative_to(EXAMPLES_DIR).as_posix() for path in EXAMPLES_DIR.rglob("*.yaml")
    }
    expected_files = {
        "models/lr.yaml",
        "models/deepfm.yaml",
        "models/mmoe.yaml",
        "models/gdcn_esmm.yaml",
        "models/esmm_output_contract.yaml",
        "models/unimixer.yaml",
        "models/token_mixer_large.yaml",
        "models/rankmixer.yaml",
        "models/rankup.yaml",
        "models/hyformer.yaml",
        "models/fat.yaml",
        "models/mixformer.yaml",
        "models/onerank.yaml",
        "models/onetrans.yaml",
        "models/pepnet.yaml",
        "shared/train_defaults.yaml",
        "shared/demo_label_policy.yaml",
        "shared/feature_config_demo.yaml",
    }

    for path in list(model_configs.values()) + ancillary_configs:
        assert path.exists(), path
    assert actual_files == expected_files

    demo_fc = FlowConfig.from_yaml(
        str(EXAMPLES_DIR / "shared" / "feature_config_demo.yaml")
    )
    assert [s.name for s in demo_fc.label_sources] == [
        "is_click",
        "is_cvr",
        "is_click_detail",
        "is_click_stock",
        "stay_time_label",
        "ctr",
        "cvr",
    ]
    assert len(demo_fc.feature_sources) == 38

    gdcn = yaml.safe_load(model_configs["demo_gdcn_esmm"].read_text(encoding="utf-8"))
    assert gdcn["type"] == "gdcn_esmm"
    assert "task_config" not in gdcn
    assert gdcn["output_contract"]["version"] == 1

    native_esmm = yaml.safe_load(
        (EXAMPLES_DIR / "models" / "esmm_output_contract.yaml").read_text(encoding="utf-8")
    )
    assert native_esmm["type"] == "esmm"
    assert "task_config" not in native_esmm
    assert native_esmm["output_contract"]["version"] == 1
    assert native_esmm["output_contract"]["objectives"][1]["loss"]["type"] == (
        "focal_binary_cross_entropy"
    )
    assert {output["name"] for output in native_esmm["output_contract"]["outputs"]} == {
        "ctr",
        "ctcvr",
        "ctdetail",
        "ctstock",
        "ctstay",
    }

    lr = yaml.safe_load(model_configs["demo_lr"].read_text(encoding="utf-8"))
    assert lr["type"] == "lr"
    assert lr["output_contract"]["outputs"] == [{"name": "pred", "source": "pred_prob"}]

    unimixer = yaml.safe_load(model_configs["demo_unimixer"].read_text(encoding="utf-8"))
    assert unimixer["type"] == "unimixer"
    assert unimixer["use_siamese"] is False
    assert "task_config" not in unimixer
    assert unimixer["output_contract"]["version"] == 1

    label_policy = yaml.safe_load(
        (EXAMPLES_DIR / "shared" / "demo_label_policy.yaml").read_text(encoding="utf-8")
    )
    assert label_policy["click"]["threshold"] == 0.42
    assert label_policy["stay_time_label"]["noise_min"] == -25


def test_lr_ctr_duplicate_config_was_removed():
    assert not (REPO_ROOT / "python" / "demo" / "model_lr_ctr.yaml").exists()


def test_serving_verification_rereads_serialized_values_as_strings(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("id,score\n001,1.5\n", encoding="utf-8")

    frame = read_serving_dataframe(path)

    assert frame.to_dict("records") == [{"id": "001", "score": "1.5"}]


def test_output_comparison_rejects_missing_prediction_columns(tmp_path):
    python_path = tmp_path / "python.csv"
    rust_path = tmp_path / "rust.csv"
    python_path.write_text("probability_ctr\n0.5\n", encoding="utf-8")
    rust_path.write_text("logit_ctr\n0.5\n", encoding="utf-8")

    success, metrics = compare_outputs(python_path, rust_path, 1e-5)

    assert success is False
    assert metrics == {}


def test_serving_array_applies_output_kind_semantics():
    logits = torch.tensor([-1.0, 0.0, 1.0])
    probabilities = torch.tensor([0.2, 0.5, 0.8])

    binary = serving_array(OutputTensor(logits, "binary_logit"))
    probability = serving_array(OutputTensor(probabilities, "probability"))

    assert np.allclose(binary, torch.sigmoid(logits).numpy())
    assert np.array_equal(probability, probabilities.numpy())


def test_demo_model_path_index_covers_all_example_models():
    assert MODEL_CONFIGS == {
        "demo_lr": EXAMPLES_DIR / "models" / "lr.yaml",
        "demo_deepfm": EXAMPLES_DIR / "models" / "deepfm.yaml",
        "demo_mmoe": EXAMPLES_DIR / "models" / "mmoe.yaml",
        "demo_esmm": EXAMPLES_DIR / "models" / "esmm_output_contract.yaml",
        "demo_gdcn_esmm": EXAMPLES_DIR / "models" / "gdcn_esmm.yaml",
        "demo_unimixer": EXAMPLES_DIR / "models" / "unimixer.yaml",
        "demo_token_mixer_large": EXAMPLES_DIR / "models" / "token_mixer_large.yaml",
        "demo_rankmixer": EXAMPLES_DIR / "models" / "rankmixer.yaml",
        "demo_rankup": EXAMPLES_DIR / "models" / "rankup.yaml",
        "demo_hyformer": EXAMPLES_DIR / "models" / "hyformer.yaml",
        "demo_fat": EXAMPLES_DIR / "models" / "fat.yaml",
        "demo_mixformer": EXAMPLES_DIR / "models" / "mixformer.yaml",
        "demo_onerank": EXAMPLES_DIR / "models" / "onerank.yaml",
        "demo_onetrans": EXAMPLES_DIR / "models" / "onetrans.yaml",
        "demo_pepnet": EXAMPLES_DIR / "models" / "pepnet.yaml",
    }


def test_verify_all_expands_and_validates_model_selection():
    assert selected_model_names("all") == list(MODEL_CONFIGS)
    assert selected_model_names("demo_lr, demo_esmm") == [
        "demo_lr",
        "demo_esmm",
    ]
    with pytest.raises(ValueError, match="unknown models: missing"):
        selected_model_names("missing")
