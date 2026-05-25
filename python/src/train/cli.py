from __future__ import annotations

"""训练脚本通用 CLI 组件。"""

import argparse
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config_train import TrainConfig
from .dag import FeatureDag
from .eval.evaluator import EvalConfig
from .manifest import write_model_manifest
from .models import ModelConfig, get_output_spec


@dataclass
class BuiltModel:
    model: torch.nn.Module
    config: ModelConfig
    spec: dict[str, Any]
    param_count: int


@dataclass
class ExportBundle:
    export_path: Path
    manifest_path: Path
    feature_config_path: Path
    model_config_path: Path
    model_version: str


def add_training_args(parser: argparse.ArgumentParser, *, lr: float, batch_size: int) -> None:
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optim", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--emb-lr", type=float)
    parser.add_argument("--emb-weight-decay", type=float)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-max-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping", type=int, default=5)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--loss-weighting", default="static", choices=["equal", "static", "uncertainty"]
    )
    parser.add_argument("--tb-dir", default="")
    parser.add_argument("--eval-metrics", default="auc")
    parser.add_argument("--monitor-metric", default="auc")
    parser.add_argument("--eval-log", default="")
    parser.add_argument("--gauc-group-feature", default="user_id")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def train_config_from_args(args: argparse.Namespace, *, export_path: str | Path) -> TrainConfig:
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        export_path=str(export_path),
        eval_samples=args.eval_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        optim={
            "name": args.optim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "emb_lr": args.emb_lr,
            "emb_weight_decay": args.emb_weight_decay,
        },
        lr_schedule={"warmup_steps": args.warmup_steps, "min_lr_ratio": args.min_lr_ratio},
        eval=EvalConfig(
            metrics=split_csv(args.eval_metrics),
            monitor_metric=args.monitor_metric,
            log_path=args.eval_log,
            gauc_group_feature=args.gauc_group_feature,
        ),
        grad_max_norm=args.grad_max_norm,
        early_stopping_patience=args.early_stopping,
        ema_decay=0.0 if args.no_ema else args.ema_decay,
        loss_weighting=args.loss_weighting,
        tb_dir=args.tb_dir,
    )


def build_model_for_dag(
    model_config_path: str | Path,
    dag: FeatureDag,
    device: torch.device,
) -> BuiltModel:
    model_config = ModelConfig.from_yaml(model_config_path)
    features = dag.feature_tuples()
    tokenizer = None
    if model_config.type == "unimixer":
        from .models.unimixer.tokenizer import FeatureTokenizer

        params = model_config.params
        tokenizer = FeatureTokenizer(
            features,
            params.get("token_dim", 64),
            params.get("num_tokens", 8),
            pooling_map=dag.feature_pooling(),
        )

    model = model_config.build(
        features,
        tokenizer=tokenizer,
        pooling_map=dag.feature_pooling(),
        total_dim=dag.feature_total_dim(),
    )
    if model_config.type == "unimixer":
        model = wrap_unimixer_for_rust_names(model)

    model = model.to(device)
    spec = get_output_spec(model_config.type, model, model_config.params)
    return BuiltModel(
        model=model,
        config=model_config,
        spec=spec,
        param_count=sum(p.numel() for p in model.parameters()),
    )


def wrap_unimixer_for_rust_names(model: torch.nn.Module) -> torch.nn.Module:
    import types
    import torch.nn as nn

    wrapper = nn.Module()
    wrapper.add_module("tokenizer", model.tokenizer)
    inner = nn.Module()
    inner.add_module("blocks", model.blocks)
    inner.add_module("task_towers", model.task_towers)
    if model.final_norm is not None:
        inner.add_module("final_norm", model.final_norm)
    wrapper.add_module("unimixer", inner)

    raw = model

    def _forward(self, x_inputs, temperature=None):
        return raw(x_inputs, temperature)

    wrapper.forward = types.MethodType(_forward, wrapper)
    return wrapper


def prepare_export_bundle(
    *,
    export_path: str | Path | None,
    export_dir: str | Path,
    model_type: str,
    feature_config_path: str | Path,
    model_config_path: str | Path,
    copy_configs: bool,
) -> ExportBundle:
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_dir = Path(export_dir if export_path is None else Path(export_path).parent)
    export_dir.mkdir(parents=True, exist_ok=True)
    weights_path = (
        Path(export_path) if export_path else export_dir / f"{model_type}_{version}.safetensors"
    )

    if copy_configs:
        feature_copy = export_dir / f"feature_config_{version}.yaml"
        model_copy = export_dir / f"model_config_{version}.yaml"
        shutil.copy(feature_config_path, feature_copy)
        shutil.copy(model_config_path, model_copy)
    else:
        feature_copy = Path(feature_config_path)
        model_copy = Path(model_config_path)

    return ExportBundle(
        export_path=weights_path,
        manifest_path=weights_path.with_suffix(".manifest.yaml"),
        feature_config_path=feature_copy,
        model_config_path=model_copy,
        model_version=version,
    )


def write_training_manifest(
    *,
    bundle: ExportBundle,
    model_id: str,
    model_type: str,
    spec: dict[str, Any],
    best_score: float,
    repo_root: str | Path,
) -> Path:
    return write_model_manifest(
        manifest_path=bundle.manifest_path,
        model_id=model_id,
        model_version=bundle.model_version,
        model_type=model_type,
        weights_path=bundle.export_path,
        feature_config_path=bundle.feature_config_path,
        model_config_path=bundle.model_config_path,
        tasks=spec["task_names"],
        label_col_map=spec["label_col_map"],
        metrics={"best_score": float(best_score)},
        repo_root=repo_root,
    )
