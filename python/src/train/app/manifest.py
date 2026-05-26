from __future__ import annotations

"""模型发布 manifest 导出。"""

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit(repo_root: str | Path | None = None) -> str:
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(
            cmd,
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def write_model_manifest(
    *,
    manifest_path: str | Path,
    model_id: str,
    model_version: str,
    model_type: str,
    weights_path: str | Path,
    feature_config_path: str | Path,
    model_config_path: str | Path,
    tasks: list[str],
    label_col_map: dict[str, str],
    metrics: dict[str, float],
    repo_root: str | Path | None = None,
    run_version: str | None = None,
    published_version: str | None = None,
    best_version: str | None = None,
    best_epoch: int | None = None,
    best_step: int | None = None,
    best_score: float | None = None,
    latest_version: str | None = None,
    latest_epoch: int | None = None,
    latest_step: int | None = None,
    checkpoint_dir: str | Path | None = None,
    run_manifest_file: str | Path | None = None,
    published_weights_file: str | Path | None = None,
    best_weights_file: str | Path | None = None,
    latest_weights_file: str | Path | None = None,
) -> Path:
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    weights_path = Path(weights_path)
    feature_config_path = Path(feature_config_path)
    model_config_path = Path(model_config_path)

    data: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "model_version": model_version,
        "run_version": run_version or model_version,
        "published_version": published_version or model_version,
        "model_type": model_type,
        "code_commit": current_git_commit(repo_root),
        "weights_file": _relative_to_manifest(weights_path, manifest_dir),
        "weights_sha256": sha256_file(weights_path),
        "feature_config_file": _relative_to_manifest(feature_config_path, manifest_dir),
        "feature_config_sha256": sha256_file(feature_config_path),
        "model_config_file": _relative_to_manifest(model_config_path, manifest_dir),
        "model_config_sha256": sha256_file(model_config_path),
        "tasks": tasks,
        "label_col_map": label_col_map,
        "metrics": metrics,
    }
    if best_version is not None:
        data["best_version"] = best_version
    if best_epoch is not None:
        data["best_epoch"] = best_epoch
    if best_step is not None:
        data["best_step"] = best_step
    if best_score is not None:
        data["best_score"] = best_score
    if latest_version is not None:
        data["latest_version"] = latest_version
    if latest_epoch is not None:
        data["latest_epoch"] = latest_epoch
    if latest_step is not None:
        data["latest_step"] = latest_step
    if checkpoint_dir is not None:
        data["checkpoint_dir"] = str(checkpoint_dir)
    if run_manifest_file is not None:
        data["run_manifest_file"] = str(run_manifest_file)
    if published_weights_file is not None:
        data["published_weights_file"] = _relative_to_manifest(
            Path(published_weights_file), manifest_dir
        )
    if best_weights_file is not None:
        data["best_weights_file"] = _relative_to_manifest(Path(best_weights_file), manifest_dir)
    if latest_weights_file is not None:
        data["latest_weights_file"] = _relative_to_manifest(Path(latest_weights_file), manifest_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return manifest_path


def _relative_to_manifest(path: Path, manifest_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(path)
