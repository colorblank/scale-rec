from __future__ import annotations

from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from train.app.artifacts import TrainingArtifactManager, load_resume_state
from train.app.export import replace_inactive_embedding_rows
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
        resume_state={
            "schema_version": 1,
            "checkpoint_kind": "epoch",
            "epoch": 1,
            "batch_in_epoch": 0,
            "next_epoch": 2,
            "global_step": 1,
            "best_score": 0.7,
            "stale_epochs": 0,
            "best_epoch": 1,
            "periodic_checkpoint_seq": 0,
            "last_periodic_checkpoint_step": 0,
        },
    )
    manager.write_embedding_bucket_report(
        {
            "schema_version": 1,
            "training_steps": 1,
            "features": {
                "user_id": {
                    "vocab_size": 2,
                    "total_hits": 2,
                    "active_buckets": 1,
                    "inactive_buckets": 1,
                    "bucket_utilization": 0.5,
                    "inactive_bucket_ids": [0],
                    "bucket_hits": [0, 2],
                }
            },
        }
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
    assert manager.paths.best_state_path.exists()
    assert manager.paths.latest_state_path.exists()
    assert manager.paths.embedding_bucket_report_path.exists()
    assert load_resume_state(manager.paths.best_alias_path)["best_score"] == 0.7

    published_data = yaml.safe_load(published.read_text(encoding="utf-8"))
    run_data = yaml.safe_load(run_manifest.read_text(encoding="utf-8"))

    assert published_data["model_id"] == "demo_model"
    assert published_data["model_version"] == "run-001"
    assert published_data["run_version"] == "run-001"
    assert published_data["published_version"] == "epoch-0001-step-000001"
    assert published_data["best_version"] == "epoch-0001-step-000001"
    assert published_data["weights_file"] == "published.safetensors"
    assert published_data["checkpoint_dir"].endswith("checkpoints")
    assert published_data["embedding_bucket_report_file"].endswith(
        "embedding_bucket_report.yaml"
    )

    assert run_data["model_name"] == "demo_model"
    assert run_data["model_version"] == "run-001"
    assert run_data["best_version"] == "epoch-0001-step-000001"
    assert run_data["feature_config_file"] == str(manager.paths.feature_config_path)
    assert run_data["model_config_file"] == str(manager.paths.model_config_path)
    assert run_data["published_source_file"].endswith("best.safetensors")
    assert run_data["embedding_bucket_report_file"] == str(
        manager.paths.embedding_bucket_report_path
    )
    assert run_data["checkpoints"][0]["state_path"].endswith("epoch-0001-step-000001.resume.pt")


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


def test_inactive_dict_mapper_default_bucket_uses_active_row_mean(tmp_path):
    path = tmp_path / "model.safetensors"
    weight = torch.tensor(
        [
            [100.0, 100.0],
            [1.0, 3.0],
            [5.0, 7.0],
            [200.0, 200.0],
        ]
    )
    save_file({"embeddings.emb_category.weight": weight}, path)

    replace_inactive_embedding_rows(
        path,
        {
            "features": {
                "category": {
                    "operator_type": "DictMapper",
                    "default_idx": 0,
                    "bucket_hits": [0, 4, 2, 0],
                    "inactive_bucket_ids": [0, 3],
                }
            }
        },
    )

    exported = load_file(path)["embeddings.emb_category.weight"]
    expected_mean = torch.tensor([3.0, 5.0])
    assert torch.equal(exported[0], expected_mean)
    assert torch.equal(exported[3], expected_mean)
    assert torch.equal(exported[1], weight[1])
    assert torch.equal(exported[2], weight[2])
