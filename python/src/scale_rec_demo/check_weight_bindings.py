from __future__ import annotations

"""Check PyTorch state_dict keys against Rust Candle VarBuilder paths."""

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from .paths import DISCOVER_FEATURE_CONFIG, MODEL_CONFIGS, REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export example PyTorch models and validate Rust weight binding."
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated MODEL_CONFIGS keys, or 'all'.",
    )
    parser.add_argument("--feature-config", default=str(DISCOVER_FEATURE_CONFIG))
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated safetensors and manifests for inspection.",
    )
    return parser.parse_args()


def selected_models(value: str) -> list[tuple[str, Path]]:
    if value == "all":
        return list(MODEL_CONFIGS.items())
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODEL_CONFIGS]
    if unknown:
        raise SystemExit(f"unknown models: {', '.join(unknown)}")
    return [(name, MODEL_CONFIGS[name]) for name in names]


def _missing_training_deps(exc: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        f"check_weight_bindings requires project training dependencies: missing {exc.name}"
    )


def build_feature_info(feature_config: Path):
    try:
        from train.core.builder import DagBuilder
        from train.core.config import FlowConfig
        from train.core.feature_info import FeatureInfo
    except ModuleNotFoundError as exc:
        raise _missing_training_deps(exc) from exc

    flow = FlowConfig.from_yaml(str(feature_config))
    artifact = DagBuilder.build(flow)
    return FeatureInfo(
        artifact.sources,
        artifact.node_defs,
        artifact.feature_schemas,
        artifact.execution_order,
    )


def export_manifest_pair(
    *,
    model_name: str,
    model_config: Path,
    feature_config: Path,
    feat_info,
    out_dir: Path,
) -> Path:
    try:
        torch = importlib.import_module("torch")
        build_model_for_dag = importlib.import_module("train.app.cli").build_model_for_dag
        export_to_safetensors = importlib.import_module("train.app.export").export_to_safetensors
        write_model_manifest = importlib.import_module("train.app.manifest").write_model_manifest
        task_specs_to_manifest = importlib.import_module(
            "train.core.task"
        ).task_specs_to_manifest
    except ModuleNotFoundError as exc:
        raise _missing_training_deps(exc) from exc

    built = build_model_for_dag(model_config, feat_info, torch.device("cpu"))
    weights_path = out_dir / f"{model_name}.safetensors"
    manifest_path = out_dir / f"{model_name}.manifest.yaml"
    export_to_safetensors(built.model, str(weights_path))
    task_specs = built.spec.get("tasks")
    write_model_manifest(
        manifest_path=manifest_path,
        model_id=model_name,
        model_version="state-dict-check",
        model_type=built.config.type,
        weights_path=weights_path,
        feature_config_path=feature_config,
        model_config_path=model_config,
        tasks=built.spec.get("task_names", []),
        label_col_map=built.spec.get("label_col_map", {}),
        metrics={},
        task_specs=task_specs_to_manifest(task_specs),
        repo_root=REPO_ROOT,
    )
    return manifest_path


def validate_with_rust(manifests: list[Path]) -> None:
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "validate_manifest",
        "--",
        *[str(path) for path in manifests],
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    feature_config = Path(args.feature_config)
    temp_dir = Path(tempfile.mkdtemp(prefix="scale-rec-weight-bindings-"))
    try:
        feat_info = build_feature_info(feature_config)
        manifests = []
        for model_name, model_config in selected_models(args.models):
            print(f"[export] {model_name}")
            manifests.append(
                export_manifest_pair(
                    model_name=model_name,
                    model_config=model_config,
                    feature_config=feature_config,
                    feat_info=feat_info,
                    out_dir=temp_dir,
                )
            )
        validate_with_rust(manifests)
        print(f"validated {len(manifests)} model weight bindings")
        if args.keep_temp:
            print(f"temp dir: {temp_dir}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
