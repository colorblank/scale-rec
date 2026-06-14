from __future__ import annotations

"""训练脚本通用 CLI 组件。"""

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..core.config import ArtifactConfig, EvalConfig, ModelConfig, TrainConfig
from ..models import get_output_spec

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
CONSOLE_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
FILE_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(process)d %(name)s:%(lineno)d: %(message)s"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "examples" / "shared" / "train_defaults.yaml"


@dataclass
class BuiltModel:
    model: torch.nn.Module
    config: ModelConfig
    spec: dict[str, Any]
    param_count: int


def add_training_args(parser: argparse.ArgumentParser, *, lr: float, batch_size: int) -> None:
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--optim", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--emb-lr", type=float)
    parser.add_argument("--emb-weight-decay", type=float)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--prefetch-batches", type=int)
    parser.add_argument("--checkpoint-interval-steps", type=int)
    parser.add_argument("--checkpoint-interval-seconds", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--grad-max-norm", type=float)
    parser.add_argument("--early-stopping", type=int)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--loss-weighting", choices=["equal", "static", "uncertainty"])
    parser.add_argument("--tb-dir")
    parser.add_argument("--eval-metrics")
    parser.add_argument("--monitor-metric")
    parser.add_argument("--monitor-task")
    parser.add_argument("--monitor-mode", choices=["auto", "max", "min"])
    parser.add_argument("--eval-log")
    parser.add_argument("--gauc-group-feature")
    parser.add_argument(
        "--init-weights",
        default="",
        help="safetensors weights used to initialize the model before fine-tuning",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="resume training from a saved checkpoint weights file or .resume.pt sidecar",
    )


def add_data_range_args(parser: argparse.ArgumentParser, *, data_required: bool) -> None:
    parser.add_argument("--data", required=data_required)
    parser.add_argument(
        "--data-glob",
        default="",
        help="glob pattern for dated training files; used before --data when set",
    )
    parser.add_argument("--start-date", default="", help="inclusive YYYYMMDD start date")
    parser.add_argument("--end-date", default="", help="inclusive YYYYMMDD end date")


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS)
    parser.add_argument("--file-log-level", default="DEBUG", choices=LOG_LEVELS)
    parser.add_argument("--log-dir", default="", help="directory for timestamped training logs")
    parser.add_argument(
        "--log-file", default="", help="explicit log file path; overrides --log-dir"
    )


def add_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-name", default="", help="logical model name for published artifacts"
    )
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


_DATE_RE = re.compile(r"(\d{8})")


def resolve_data_paths(args: argparse.Namespace) -> list[str]:
    """Resolve CLI data arguments into an ordered, non-empty file list."""

    pattern = getattr(args, "data_glob", "")
    if pattern:
        return _resolve_globbed_data_paths(
            pattern,
            start_date=getattr(args, "start_date", ""),
            end_date=getattr(args, "end_date", ""),
        )

    data = getattr(args, "data", None)
    if not data:
        raise SystemExit("--data is required unless --data-glob is provided")
    return [str(data)]


def _resolve_globbed_data_paths(pattern: str, *, start_date: str, end_date: str) -> list[str]:
    if not start_date or not end_date:
        raise SystemExit("--start-date and --end-date are required with --data-glob")

    start = _parse_yyyymmdd(start_date, "--start-date")
    end = _parse_yyyymmdd(end_date, "--end-date")
    if start > end:
        raise SystemExit("--start-date must be <= --end-date")

    by_date: dict[str, list[str]] = {}
    pattern_path = Path(pattern)
    search_root = pattern_path.parent if pattern_path.parent != Path() else Path()
    for matched_path in search_root.glob(pattern_path.name):
        path = str(matched_path)
        date_text = _extract_basename_date(matched_path)
        if date_text is None:
            continue
        date = _parse_yyyymmdd(date_text, Path(path).name)
        if start <= date <= end:
            by_date.setdefault(date_text, []).append(path)

    expected = _date_range_strings(start, end)
    missing = [date for date in expected if date not in by_date]
    if missing:
        raise SystemExit("missing data files for dates: " + ", ".join(missing))

    resolved: list[str] = []
    for date in expected:
        resolved.extend(sorted(by_date[date]))
    if not resolved:
        raise SystemExit(f"--data-glob matched no dated files in range: {pattern}")
    return resolved


def _extract_basename_date(path: Path) -> str | None:
    match = _DATE_RE.search(path.name)
    return match.group(1) if match else None


def _parse_yyyymmdd(value: str, label: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SystemExit(f"{label} must be YYYYMMDD: {value}") from exc


def _date_range_strings(start: datetime, end: datetime) -> list[str]:
    days = (end - start).days
    return [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days + 1)]


def describe_data_paths(paths: list[str]) -> str:
    dates = [_extract_basename_date(Path(path)) for path in paths]
    dates = [date for date in dates if date is not None]
    if dates:
        return f"{len(paths)} files date_range={min(dates)}..{max(dates)}"
    return f"{len(paths)} files"


def load_init_weights(
    model: torch.nn.Module,
    init_weights: str | Path | None,
    device: torch.device,
) -> None:
    """Load safetensors model weights for fine-tuning, without optimizer state."""

    if not init_weights:
        return
    path = Path(init_weights)
    if not path.exists():
        raise FileNotFoundError(f"--init-weights not found: {path}")
    state = load_file(str(path))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"failed to load --init-weights {path}: {exc}") from exc
    model.to(device)
    logging.getLogger(__name__).info("initialized model from %s", path)


def train_config_from_args(args: argparse.Namespace, *, export_path: str | Path) -> TrainConfig:
    config_path = Path(getattr(args, "train_config", DEFAULT_TRAIN_CONFIG))
    base = TrainConfig.from_yaml(config_path) if config_path.exists() else TrainConfig()

    def pick(name: str, fallback: Any) -> Any:
        value = getattr(args, name, None)
        return fallback if value is None else value

    eval_metrics = pick("eval_metrics", ",".join(base.eval.metrics))
    if isinstance(eval_metrics, str):
        eval_metrics = split_csv(eval_metrics)

    return TrainConfig(
        epochs=pick("epochs", base.epochs),
        batch_size=pick("batch_size", base.batch_size),
        export_path=str(export_path),
        prefetch_batches=pick("prefetch_batches", base.prefetch_batches),
        checkpoint_interval_steps=pick("checkpoint_interval_steps", base.checkpoint_interval_steps),
        checkpoint_interval_seconds=pick(
            "checkpoint_interval_seconds", base.checkpoint_interval_seconds
        ),
        artifacts=ArtifactConfig(
            artifact_root=str(pick("artifact_dir", base.artifacts.artifact_root)),
            model_name=str(pick("model_name", base.artifacts.model_name)),
            run_version=str(pick("run_version", base.artifacts.run_version)),
            keep_checkpoints=int(pick("keep_checkpoints", base.artifacts.keep_checkpoints)),
            publish_best=base.artifacts.publish_best,
            publish_latest=base.artifacts.publish_latest,
            copy_configs=base.artifacts.copy_configs,
        ),
        eval_samples=pick("eval_samples", base.eval_samples),
        eval_interval=pick("eval_interval", base.eval_interval),
        log_interval=pick("log_interval", base.log_interval),
        optim={
            "name": pick("optim", base.optim.name),
            "lr": pick("lr", base.optim.lr),
            "weight_decay": pick("weight_decay", base.optim.weight_decay),
            "emb_lr": pick("emb_lr", base.optim.emb_lr),
            "emb_weight_decay": pick("emb_weight_decay", base.optim.emb_weight_decay),
        },
        lr_schedule={
            "warmup_steps": pick("warmup_steps", base.lr_schedule.warmup_steps),
            "min_lr_ratio": pick("min_lr_ratio", base.lr_schedule.min_lr_ratio),
        },
        eval=EvalConfig(
            metrics=eval_metrics,
            monitor_metric=pick("monitor_metric", base.eval.monitor_metric),
            monitor_task=pick("monitor_task", base.eval.monitor_task),
            monitor_mode=pick("monitor_mode", base.eval.monitor_mode),
            log_path=pick("eval_log", base.eval.log_path),
            gauc_group_feature=pick("gauc_group_feature", base.eval.gauc_group_feature),
        ),
        grad_max_norm=pick("grad_max_norm", base.grad_max_norm),
        early_stopping_patience=pick("early_stopping", base.early_stopping_patience),
        ema_decay=0.0 if getattr(args, "no_ema", False) else pick("ema_decay", base.ema_decay),
        loss_weighting=pick("loss_weighting", base.loss_weighting),
        tb_dir=pick("tb_dir", base.tb_dir),
    )


def build_model_for_dag(
    model_config_path: str | Path,
    feat_info,
    device: torch.device,
) -> BuiltModel:
    model_config = ModelConfig.from_yaml(model_config_path)
    features = feat_info.feature_tuples()
    tokenizer = None
    if model_config.type == "unimixer":
        from ..models.unimixer.tokenizer import FeatureTokenizer

        params = model_config.params
        tokenizer = FeatureTokenizer(
            features,
            params.get("token_dim", 64),
            params.get("num_tokens", 8),
            pooling_map=feat_info.feature_pooling(),
            seq_len_map=feat_info.feature_seq_lens(),
        )

    model = model_config.build(
        features,
        tokenizer=tokenizer,
        pooling_map=feat_info.feature_pooling(),
        total_dim=feat_info.feature_total_dim(),
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
