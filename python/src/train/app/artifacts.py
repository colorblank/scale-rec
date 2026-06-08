from __future__ import annotations

"""训练产物与 checkpoint 管理。"""

import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ..core.config import ArtifactConfig
from .export import export_to_safetensors
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

    def checkpoint_path(self, version: str) -> Path:
        return self.checkpoints_dir / f"{version}.safetensors"


@dataclass
class TrainingArtifactManager:
    paths: TrainingArtifactPaths
    keep_checkpoints: int = 3
    publish_best_alias: bool = True
    publish_latest_alias: bool = True
    copy_configs: bool = True
    _history: list[CheckpointRecord] = field(default_factory=list, init=False)
    _best: Optional[CheckpointRecord] = field(default=None, init=False)
    _latest: Optional[CheckpointRecord] = field(default=None, init=False)

    @classmethod
    def from_config(
        cls,
        artifact_config: ArtifactConfig,
        *,
        model_name: str,
        model_type: str,
        artifact_root: Union[str, Path],
        publish_path: Union[str, Path, None],
        feature_config_path: Union[str, Path],
        model_config_path: Union[str, Path],
    ) -> "TrainingArtifactManager":
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

    def prepare(
        self, feature_config_path: Union[str, Path], model_config_path: Union[str, Path]
    ) -> None:
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
        version: Optional[str] = None,
    ) -> CheckpointRecord:
        version = version or f"epoch-{epoch:04d}-step-{step:06d}"
        path = self.paths.checkpoint_path(version)
        export_to_safetensors(model, path)
        record = CheckpointRecord(
            version=version,
            epoch=epoch,
            step=step,
            score=score,
            metric_name=metric_name,
            path=path,
        )
        self._history.append(record)
        self._latest = record
        if self.publish_latest_alias:
            shutil.copy2(path, self.paths.latest_alias_path)
        if is_best:
            self._best = record
            if self.publish_best_alias:
                shutil.copy2(path, self.paths.best_alias_path)
        self._prune_history()
        return record

    @property
    def history(self) -> list[CheckpointRecord]:
        return list(self._history)

    @property
    def best(self) -> Optional[CheckpointRecord]:
        return self._best

    @property
    def latest(self) -> Optional[CheckpointRecord]:
        return self._latest

    def finalize(
        self,
        *,
        model: Optional[Any],
        model_type: str,
        tasks: list[str],
        label_col_map: dict[str, str],
        metrics: dict[str, float],
        repo_root: Union[str, Path, None],
        published_version: Optional[str] = None,
        best_score: Optional[float] = None,
        published_source: Union[str, Path, None] = None,
    ) -> None:
        self.paths.published_weights_path.parent.mkdir(parents=True, exist_ok=True)
        if published_source is not None:
            shutil.copy2(published_source, self.paths.published_weights_path)
        elif model is not None:
            export_to_safetensors(model, self.paths.published_weights_path)
        self._write_run_manifest(
            model_type=model_type,
            best_score=best_score,
            published_version=published_version,
            published_source=published_source,
        )
        self._write_published_manifest(
            model_type=model_type,
            tasks=tasks,
            label_col_map=label_col_map,
            metrics=metrics,
            repo_root=repo_root,
            published_version=published_version,
            best_score=best_score,
        )

    def _write_run_manifest(
        self,
        *,
        model_type: str,
        best_score: Optional[float],
        published_version: Optional[str],
        published_source: Union[str, Path, None],
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
                asdict(record) | {"path": str(record.path)} for record in self._history
            ],
        }
        self.paths.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths.run_manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def _write_published_manifest(
        self,
        *,
        model_type: str,
        tasks: list[str],
        label_col_map: dict[str, str],
        metrics: dict[str, float],
        repo_root: Union[str, Path, None],
        published_version: Optional[str],
        best_score: Optional[float],
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
        )

    def _prune_history(self) -> None:
        if self.keep_checkpoints <= 0:
            return
        while len(self._history) > self.keep_checkpoints:
            record = self._history.pop(0)
            if record.path.exists():
                record.path.unlink()
