from __future__ import annotations

import yaml

from train.manifest import sha256_file, write_model_manifest


def test_write_model_manifest_records_relative_files_and_hashes(tmp_path):
    weights = tmp_path / "model.safetensors"
    feature_config = tmp_path / "feature.yaml"
    model_config = tmp_path / "model.yaml"
    manifest = tmp_path / "model.manifest.yaml"
    weights.write_bytes(b"weights")
    feature_config.write_text("sources: []\noperators: []\n", encoding="utf-8")
    model_config.write_text("type: lr\n", encoding="utf-8")

    write_model_manifest(
        manifest_path=manifest,
        model_id="model",
        model_version="v1",
        model_type="lr",
        weights_path=weights,
        feature_config_path=feature_config,
        model_config_path=model_config,
        tasks=["pred"],
        label_col_map={"pred": "is_click"},
        metrics={"best_auc": 0.5},
    )

    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["weights_file"] == "model.safetensors"
    assert data["feature_config_sha256"] == sha256_file(feature_config)
    assert data["model_config_sha256"] == sha256_file(model_config)
    assert data["tasks"] == ["pred"]
    assert data["metrics"] == {"best_auc": 0.5}
