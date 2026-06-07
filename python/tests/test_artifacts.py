from __future__ import annotations

from pathlib import Path

import torch
import yaml

from train.app.artifacts import TrainingArtifactManager
from train.core.config import ArtifactConfig


def test_training_artifacts_manage_run_best_and_published_versions(tmp_path):
    feature_config = tmp_path / "feature.yaml"
    model_config = tmp_path / "model.yaml"
    feature_config.write_text("sources: []\noperators: []\n", encoding="utf-8")
    model_config.write_text("type: lr\n", encoding="utf-8")

    manager = TrainingArtifactManager.from_config(
        ArtifactConfig(
            artifact_root=str(tmp_path / "artifacts"),
            model_name="demo_model",
            run_version="run-001",
            keep_checkpoints=2,
        ),
        model_name="lr",
        model_type="lr",
        artifact_root=tmp_path / "fallback",
        publish_path=tmp_path / "published.safetensors",
        feature_config_path=feature_config,
        model_config_path=model_config,
    )
    manager.prepare(feature_config, model_config)
    assert manager.paths.feature_config_path == manager.paths.configs_dir / "feature_config.yaml"
    assert manager.paths.model_config_path == manager.paths.configs_dir / "model_config.yaml"
    assert manager.paths.feature_config_path.exists()
    assert manager.paths.model_config_path.exists()

    model = torch.nn.Linear(1, 1)
    manager.save_checkpoint(
        model,
        epoch=1,
        step=1,
        score=0.7,
        metric_name="auc",
        is_best=True,
    )
    manager.finalize(
        model=None,
        model_type="lr",
        tasks=["pred"],
        label_col_map={"pred": "is_click"},
        metrics={"best_score": 0.7},
        repo_root=tmp_path,
        published_version="epoch-0001-step-000001",
        best_score=0.7,
        published_source=manager.paths.best_alias_path,
    )

    published = manager.paths.published_manifest_path
    run_manifest = manager.paths.run_manifest_path
    assert published.exists()
    assert run_manifest.exists()
    assert manager.paths.published_weights_path.exists()
    assert manager.paths.best_alias_path.exists()
    assert manager.paths.latest_alias_path.exists()

    published_data = yaml.safe_load(published.read_text(encoding="utf-8"))
    run_data = yaml.safe_load(run_manifest.read_text(encoding="utf-8"))

    assert published_data["model_id"] == "demo_model"
    assert published_data["model_version"] == "run-001"
    assert published_data["run_version"] == "run-001"
    assert published_data["published_version"] == "epoch-0001-step-000001"
    assert published_data["best_version"] == "epoch-0001-step-000001"
    assert published_data["weights_file"] == "published.safetensors"
    assert published_data["checkpoint_dir"].endswith("checkpoints")

    assert run_data["model_name"] == "demo_model"
    assert run_data["model_version"] == "run-001"
    assert run_data["best_version"] == "epoch-0001-step-000001"
    assert run_data["feature_config_file"] == str(manager.paths.feature_config_path)
    assert run_data["model_config_file"] == str(manager.paths.model_config_path)
    assert run_data["published_source_file"].endswith("best.safetensors")


def test_training_artifacts_publish_into_run_serving_dir_by_default(tmp_path):
    feature_config = tmp_path / "feature.yaml"
    model_config = tmp_path / "model.yaml"
    feature_config.write_text("sources: []\noperators: []\n", encoding="utf-8")
    model_config.write_text("type: lr\n", encoding="utf-8")

    manager = TrainingArtifactManager.from_config(
        ArtifactConfig(
            artifact_root=str(tmp_path / "artifacts"),
            model_name="demo_model",
            run_version="run-001",
        ),
        model_name="lr",
        model_type="lr",
        artifact_root=tmp_path / "fallback",
        publish_path=None,
        feature_config_path=feature_config,
        model_config_path=model_config,
    )
    manager.prepare(feature_config, model_config)

    assert manager.paths.published_weights_path == (
        tmp_path / "artifacts" / "demo_model" / "run-001" / "serving" / "model.safetensors"
    )
    assert manager.paths.published_manifest_path == (
        tmp_path / "artifacts" / "demo_model" / "run-001" / "serving" / "model.manifest.yaml"
    )
    assert manager.paths.feature_config_path == (
        tmp_path
        / "artifacts"
        / "demo_model"
        / "run-001"
        / "serving"
        / "configs"
        / "feature_config.yaml"
    )
    assert manager.paths.model_config_path == (
        tmp_path
        / "artifacts"
        / "demo_model"
        / "run-001"
        / "serving"
        / "configs"
        / "model_config.yaml"
    )

    model = torch.nn.Linear(1, 1)
    manager.finalize(
        model=model,
        model_type="lr",
        tasks=["pred"],
        label_col_map={"pred": "is_click"},
        metrics={},
        repo_root=tmp_path,
    )

    data = yaml.safe_load(manager.paths.published_manifest_path.read_text(encoding="utf-8"))
    assert data["weights_file"] == "model.safetensors"
    assert Path(data["feature_config_file"]) == Path("configs") / "feature_config.yaml"
    assert Path(data["model_config_file"]) == Path("configs") / "model_config.yaml"
