from __future__ import annotations

"""训练产物与 checkpoint 管理。"""

import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from ..core.config import ArtifactConfig
from ..core.task import TaskSpec, task_specs_to_manifest
from ..training.quality import write_embedding_bucket_report
from .export import export_to_safetensors, replace_inactive_embedding_rows
from .manifest import write_model_manifest


def _utc_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_name(value: str) -> str:
    value = value.strip()
    if not value:
        return "model"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


@dataclass(frozen=True)
class CheckpointRecord:
    version: str
    epoch: int
    step: int
    score: float
    metric_name: str
    path: Path
    state_path: Path | None = None


@dataclass
class TrainingArtifactPaths:
    artifact_root: Path
    model_name: str
    run_version: str
    run_dir: Path
    configs_dir: Path
    checkpoints_dir: Path
    published_weights_path: Path
    published_manifest_path: Path
    run_manifest_path: Path
    feature_config_path: Path
    model_config_path: Path

    @property
    def latest_alias_path(self) -> Path:
        return self.run_dir / "latest.safetensors"

    @property
    def best_alias_path(self) -> Path:
        return self.run_dir / "best.safetensors"

    @property
    def latest_state_path(self) -> Path:
        return self.run_dir / "latest.resume.pt"

    @property
    def best_state_path(self) -> Path:
        return self.run_dir / "best.resume.pt"

    @property
    def embedding_bucket_report_path(self) -> Path:
        return self.run_dir / "embedding_bucket_report.yaml"

    def checkpoint_path(self, version: str) -> Path:
        return self.checkpoints_dir / f"{version}.safetensors"

    def checkpoint_state_path(self, version: str) -> Path:
        return self.checkpoints_dir / f"{version}.resume.pt"


@dataclass
class TrainingArtifactManager:
    paths: TrainingArtifactPaths
    keep_checkpoints: int = 3
    publish_best_alias: bool = True
    publish_latest_alias: bool = True
    copy_configs: bool = True
    _history: list[CheckpointRecord] = field(default_factory=list, init=False)
    _best: CheckpointRecord | None = field(default=None, init=False)
    _latest: CheckpointRecord | None = field(default=None, init=False)

    @classmethod
    def from_config(
        cls,
        artifact_config: ArtifactConfig,
        *,
        model_name: str,
        model_type: str,
        artifact_root: str | Path,
        publish_path: str | Path | None,
        feature_config_path: str | Path,
        model_config_path: str | Path,
    ) -> TrainingArtifactManager:
        resolved_model_name = _safe_name(artifact_config.model_name or model_name or model_type)
        run_version = artifact_config.run_version or _utc_version()
        root = Path(artifact_config.artifact_root or artifact_root)
        run_dir = root / resolved_model_name / run_version
        checkpoints_dir = run_dir / "checkpoints"
        serving_dir = run_dir / "serving"
        configs_dir = serving_dir / "configs"
        if publish_path:
            published_weights_path = Path(publish_path)
        else:
            published_weights_path = serving_dir / "model.safetensors"
        published_manifest_path = published_weights_path.with_suffix(".manifest.yaml")
        feature_copy_path = configs_dir / "feature_config.yaml"
        model_copy_path = configs_dir / "model_config.yaml"
        paths = TrainingArtifactPaths(
            artifact_root=root,
            model_name=resolved_model_name,
            run_version=run_version,
            run_dir=run_dir,
            configs_dir=configs_dir,
            checkpoints_dir=checkpoints_dir,
            published_weights_path=published_weights_path,
            published_manifest_path=published_manifest_path,
            run_manifest_path=run_dir / "run.manifest.yaml",
            feature_config_path=feature_copy_path,
            model_config_path=model_copy_path,
        )
        return cls(
            paths=paths,
            keep_checkpoints=artifact_config.keep_checkpoints,
            publish_best_alias=artifact_config.publish_best,
            publish_latest_alias=artifact_config.publish_latest,
            copy_configs=artifact_config.copy_configs,
        )

    def prepare(self, feature_config_path: str | Path, model_config_path: str | Path) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if self.copy_configs:
            self.paths.configs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(feature_config_path, self.paths.feature_config_path)
            shutil.copy2(model_config_path, self.paths.model_config_path)
        else:
            self.paths.feature_config_path = Path(feature_config_path)
            self.paths.model_config_path = Path(model_config_path)

    def save_checkpoint(
        self,
        model: Any,
        *,
        epoch: int,
        step: int,
        score: float,
        metric_name: str,
        is_best: bool,
        resume_state: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> CheckpointRecord:
        version = version or f"epoch-{epoch:04d}-step-{step:06d}"
        path = self.paths.checkpoint_path(version)
        state_path = self.paths.checkpoint_state_path(version) if resume_state is not None else None
        export_to_safetensors(model, path)
        if resume_state is not None:
            self._write_resume_state(state_path, resume_state)
        record = CheckpointRecord(
            version=version,
            epoch=epoch,
            step=step,
            score=score,
            metric_name=metric_name,
            path=path,
            state_path=state_path,
        )
        self._history.append(record)
        self._latest = record
        if self.publish_latest_alias:
            shutil.copy2(path, self.paths.latest_alias_path)
            if state_path is not None:
                shutil.copy2(state_path, self.paths.latest_state_path)
        if is_best:
            self._best = record
            if self.publish_best_alias:
                shutil.copy2(path, self.paths.best_alias_path)
                if state_path is not None:
                    shutil.copy2(state_path, self.paths.best_state_path)
        self._prune_history()
        return record

    @property
    def history(self) -> list[CheckpointRecord]:
        return list(self._history)

    @property
    def best(self) -> CheckpointRecord | None:
        return self._best

    @property
    def latest(self) -> CheckpointRecord | None:
        return self._latest

    def finalize(
        self,
        *,
        model: Any | None,
        model_type: str,
        tasks: list[str],
        label_col_map: dict[str, str],
        metrics: dict[str, float],
        repo_root: str | Path | None,
        task_specs: list[TaskSpec] | None = None,
        published_version: str | None = None,
        best_score: float | None = None,
        published_source: str | Path | None = None,
        embedding_bucket_report: dict[str, Any] | None = None,
    ) -> None:
        self.paths.published_weights_path.parent.mkdir(parents=True, exist_ok=True)
        if published_source is not None:
            shutil.copy2(published_source, self.paths.published_weights_path)
        elif model is not None:
            export_to_safetensors(model, self.paths.published_weights_path)
        if embedding_bucket_report is not None:
            self.write_embedding_bucket_report(embedding_bucket_report)
            replace_inactive_embedding_rows(
                self.paths.published_weights_path, embedding_bucket_report
            )
        self._write_run_manifest(
            model_type=model_type,
            best_score=best_score,
            published_version=published_version,
            published_source=published_source,
        )
        self._write_published_manifest(
            model_type=model_type,
            tasks=tasks,
            task_specs=task_specs,
            label_col_map=label_col_map,
            metrics=metrics,
            repo_root=repo_root,
            published_version=published_version,
            best_score=best_score,
        )

    def write_embedding_bucket_report(self, report: dict[str, Any]) -> Path:
        return write_embedding_bucket_report(report, self.paths.embedding_bucket_report_path)

    def _write_run_manifest(
        self,
        *,
        model_type: str,
        best_score: float | None,
        published_version: str | None,
        published_source: str | Path | None,
    ) -> None:
        data: dict[str, Any] = {
            "schema_version": 1,
            "model_name": self.paths.model_name,
            "model_type": model_type,
            "model_version": self.paths.run_version,
            "published_version": published_version or self.paths.run_version,
            "best_version": self._best.version if self._best else "",
            "best_epoch": self._best.epoch if self._best else None,
            "best_step": self._best.step if self._best else None,
            "best_score": self._best.score if self._best else best_score,
            "latest_version": self._latest.version if self._latest else "",
            "latest_epoch": self._latest.epoch if self._latest else None,
            "latest_step": self._latest.step if self._latest else None,
            "checkpoints_dir": str(self.paths.checkpoints_dir),
            "best_checkpoint_file": str(self.paths.best_alias_path),
            "latest_checkpoint_file": str(self.paths.latest_alias_path),
            "published_weights_file": str(self.paths.published_weights_path),
            "published_manifest_file": str(self.paths.published_manifest_path),
            "published_source_file": str(published_source) if published_source is not None else "",
            "run_dir": str(self.paths.run_dir),
            "artifact_root": str(self.paths.artifact_root),
            "feature_config_file": str(self.paths.feature_config_path),
            "model_config_file": str(self.paths.model_config_path),
            "checkpoints": [
                asdict(record)
                | {
                    "path": str(record.path),
                    "state_path": str(record.state_path) if record.state_path else "",
                }
                for record in self._history
            ],
        }
        if self.paths.embedding_bucket_report_path.exists():
            data["embedding_bucket_report_file"] = str(self.paths.embedding_bucket_report_path)
        self.paths.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.run_manifest_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def _write_published_manifest(
        self,
        *,
        model_type: str,
        tasks: list[str],
        task_specs: list[TaskSpec] | None,
        label_col_map: dict[str, str],
        metrics: dict[str, float],
        repo_root: str | Path | None,
        published_version: str | None,
        best_score: float | None,
    ) -> None:
        manifest_metrics = dict(metrics)
        if best_score is not None:
            manifest_metrics.setdefault("best_score", float(best_score))
        write_model_manifest(
            manifest_path=self.paths.published_manifest_path,
            model_id=self.paths.model_name,
            model_version=self.paths.run_version,
            model_type=model_type,
            weights_path=self.paths.published_weights_path,
            feature_config_path=self.paths.feature_config_path,
            model_config_path=self.paths.model_config_path,
            tasks=tasks,
            task_specs=task_specs_to_manifest(task_specs),
            label_col_map=label_col_map,
            metrics=manifest_metrics,
            repo_root=repo_root,
            run_version=self.paths.run_version,
            published_version=published_version or self.paths.run_version,
            best_version=self._best.version if self._best else "",
            best_epoch=self._best.epoch if self._best else None,
            best_step=self._best.step if self._best else None,
            best_score=self._best.score if self._best else best_score,
            latest_version=self._latest.version if self._latest else "",
            latest_epoch=self._latest.epoch if self._latest else None,
            latest_step=self._latest.step if self._latest else None,
            checkpoint_dir=str(self.paths.checkpoints_dir),
            run_manifest_file=str(self.paths.run_manifest_path),
            published_weights_file=str(self.paths.published_weights_path),
            best_weights_file=str(self.paths.best_alias_path),
            latest_weights_file=str(self.paths.latest_alias_path),
            embedding_bucket_report_file=(
                str(self.paths.embedding_bucket_report_path)
                if self.paths.embedding_bucket_report_path.exists()
                else None
            ),
        )

    def _prune_history(self) -> None:
        if self.keep_checkpoints <= 0:
            return
        while len(self._history) > self.keep_checkpoints:
            record = self._history.pop(0)
            if record.path.exists():
                record.path.unlink()
            if record.state_path is not None and record.state_path.exists():
                record.state_path.unlink()

    def _write_resume_state(self, path: Path, resume_state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(resume_state, path)


def resume_state_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.name.endswith(".resume.pt"):
        return path
    if path.suffix != ".safetensors":
        raise ValueError(f"resume checkpoint must be a .safetensors file: {path}")
    return path.with_name(path.stem + ".resume.pt")


def checkpoint_weights_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.suffix == ".safetensors":
        return path
    if path.name.endswith(".resume.pt"):
        return path.with_name(path.name.removesuffix(".resume.pt") + ".safetensors")
    raise ValueError(f"checkpoint path must be a .safetensors or .resume.pt file: {path}")


def load_resume_state(checkpoint_path: str | Path) -> dict[str, Any]:
    path = resume_state_path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"resume state not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"invalid resume state file: {path}")
    return state
