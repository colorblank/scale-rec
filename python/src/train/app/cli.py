from __future__ import annotations

"""训练脚本通用 CLI 组件。"""

import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..core.config import ArtifactConfig, EvalConfig, ModelConfig, TrainConfig
from ..core.dag import FeatureDag
from ..models import get_output_spec

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
CONSOLE_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
FILE_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(process)d %(name)s:%(lineno)d: %(message)s"


@dataclass
class BuiltModel:
    model: torch.nn.Module
    config: ModelConfig
    spec: dict[str, Any]
    param_count: int


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
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS)
    parser.add_argument("--file-log-level", default="DEBUG", choices=LOG_LEVELS)
    parser.add_argument("--log-dir", default="", help="directory for timestamped training logs")
    parser.add_argument(
        "--log-file", default="", help="explicit log file path; overrides --log-dir"
    )


def add_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default="", help="logical model name for published artifacts")
    parser.add_argument("--run-version", default="", help="version string for this training run")
    parser.add_argument("--keep-checkpoints", type=int, default=3)


def configure_logging(
    level: str,
    *,
    file_level: str = "DEBUG",
    log_dir: str | Path | None = None,
    log_file: str | Path | None = None,
    run_name: str = "train",
) -> Path | None:
    """Configure console and optional file logging for training commands."""

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_parse_log_level(level))
    console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console_handler)

    resolved_log_file = _resolve_log_file(log_file=log_file, log_dir=log_dir, run_name=run_name)
    if resolved_log_file is not None:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setLevel(_parse_log_level(file_level))
        file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(file_handler)

    logging.captureWarnings(True)
    for logger_name in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if resolved_log_file is not None:
        logging.getLogger(__name__).info("log file: %s", resolved_log_file)
    return resolved_log_file


def _parse_log_level(level: str) -> int:
    try:
        return getattr(logging, level.upper())
    except AttributeError as exc:
        raise ValueError(f"unknown log level: {level}") from exc


def _resolve_log_file(
    *,
    log_file: str | Path | None,
    log_dir: str | Path | None,
    run_name: str,
) -> Path | None:
    if log_file:
        return Path(log_file)
    if not log_dir:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_name)
    return Path(log_dir) / f"{safe_name}_{timestamp}.log"


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
        artifacts=ArtifactConfig(
            artifact_root=str(getattr(args, "artifact_dir", "")),
            model_name=str(getattr(args, "model_name", "")),
            run_version=str(getattr(args, "run_version", "")),
            keep_checkpoints=int(getattr(args, "keep_checkpoints", 3)),
        ),
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
        from ..models.unimixer.tokenizer import FeatureTokenizer

        params = model_config.params
        tokenizer = FeatureTokenizer(
            features,
            params.get("token_dim", 64),
            params.get("num_tokens", 8),
            pooling_map=dag.feature_pooling(),
            seq_len_map=dag.feature_seq_lens(),
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
    wrapper.embed_dim = model.embed_dim
    wrapper.use_siamese = model.use_siamese
    wrapper.temperature = model.temperature
    wrapper.add_module("tokenizer", model.tokenizer)
    inner = nn.Module()
    inner.add_module("blocks", model.blocks)
    inner.add_module("task_towers", model.task_towers)
    if model.final_norm is not None:
        inner.add_module("final_norm", model.final_norm)
    wrapper.add_module("unimixer", inner)

    def _forward(
        self: torch.nn.Module,
        x_inputs: dict[str, torch.Tensor],
        temperature: float | None = None,
    ) -> dict[str, torch.Tensor]:
        t = temperature if temperature is not None else self.temperature
        if t <= 0:
            raise ValueError("temperature must be > 0")
        tokens = self.tokenizer(x_inputs)
        batch_size = tokens.shape[0]
        x = tokens.reshape(batch_size, self.embed_dim)
        if self.use_siamese:
            x_bar = y_bar = x
            for block in self.unimixer.blocks:
                _, x_bar, y_bar = block(x, t, x_bar, y_bar)
                x = x_bar
            output = self.unimixer.final_norm(x_bar, y_bar, None)
        else:
            for block in self.unimixer.blocks:
                x = block(x, t)
            output = x
        return self.unimixer.task_towers(output)

    wrapper.forward = types.MethodType(_forward, wrapper)
    return wrapper
