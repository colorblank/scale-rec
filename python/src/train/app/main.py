from __future__ import annotations

"""统一训练入口。

一个文件覆盖三种场景：
1. `single`：单模型训练，适合 LR / DeepFM / ESMM / UniMixer / GDCN+ESMM
2. `demo`：demo-main-sort 训练，使用单文件 TSV
3. `all`：同一数据集上批量训练多个模型
"""
import argparse
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ..core.config import FlowConfig
from ..core.dag import FeatureDag
from ..core.feature_info import FeatureInfo
from ..core.model_output import ensure_model_output
from ..core.output_contract import NormalizedOutputContract
from ..core.preprocessor import TrainingPreprocessor
from ..training.loss.objective import ObjectiveEngine, evaluate_mask
from ..training.metrics import compute_metrics
from ..training.trainer import (
    Trainer,
    build_resume_state,
    iter_preprocessed_batches,
    restore_rng_state,
)
from .artifacts import (
    TrainingArtifactManager,
    checkpoint_weights_path,
    load_resume_state,
)
from .cli import (
    add_artifact_args,
    add_data_range_args,
    add_runtime_args,
    add_training_args,
    build_model_for_dag,
    configure_logging,
    describe_data_paths,
    load_init_weights,
    resolve_data_paths,
    resolve_device,
    train_config_from_args,
)
from .data import validate_matching_text_format

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLES_DIR = REPO_ROOT / "examples"
SHARED_EXAMPLES_DIR = EXAMPLES_DIR / "shared"
MODEL_EXAMPLES_DIR = EXAMPLES_DIR / "models"
DEMO_ARTIFACT_DIR = REPO_ROOT / "python" / "artifacts" / "demo"

NULL_MARKERS: set[str] = {"NULL", "\\N", "null", "None", ""}

logger = logging.getLogger("train")


@dataclass
class PeriodicCheckpointState:
    last_step: int = 0
    last_time: float = field(default_factory=time.perf_counter)
    seq: int = 0


def _load_resume_checkpoint(checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    weights_path = checkpoint_weights_path(path)
    model_state = load_file(str(weights_path))
    resume_state = load_resume_state(path)
    return model_state, resume_state


def _restore_model_and_state(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    model_state, resume_state = _load_resume_checkpoint(checkpoint_path)
    model.load_state_dict(model_state, strict=True)
    rng_state = resume_state.get("rng_state")
    if isinstance(rng_state, dict):
        restore_rng_state(rng_state)
    return resume_state


def _wrap_unimixer(model):
    """Wrap UniMixer so state_dict matches Rust vb.pp("unimixer") prefix."""
    import types

    import torch.nn as nn

    blocks = model.blocks
    task_towers = model.task_towers
    final_norm = model.final_norm
    tokenizer = model.tokenizer

    wrapper = nn.Module()
    wrapper.add_module("tokenizer", tokenizer)
    inner = nn.Module()
    inner.add_module("blocks", blocks)
    inner.add_module("task_towers", task_towers)
    if final_norm is not None:
        inner.add_module("final_norm", final_norm)
    wrapper.add_module("unimixer", inner)

    def forward(self, x_inputs, temperature=None):
        t = temperature if temperature is not None else model.temperature
        tokens = self.tokenizer(x_inputs)
        bs = tokens.shape[0]
        x = tokens.reshape(bs, model.embed_dim)
        if model.use_siamese:
            x_bar = y_bar = x
            for blk in self.unimixer.blocks:
                _, xbn, ybn = blk(x, t, x_bar, y_bar, use_siamese=True)
                x_bar = xbn
                y_bar = ybn
                x = x_bar
            output = self.unimixer.final_norm(x_bar, y_bar, None)
        else:
            for blk in self.unimixer.blocks:
                x = blk(x, t, use_siamese=False)
            output = x
        return self.unimixer.task_towers(output)

    wrapper.forward = types.MethodType(forward, wrapper)
    return wrapper


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_dataframes(paths: list[str]) -> pd.DataFrame:
    dfs = [_load_dataframe(path) for path in paths]
    if not dfs:
        raise ValueError("no data files to load")
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


def _load_eval_dataframe(
    training_path: str,
    training_columns: list[str],
    eval_path: str,
) -> pd.DataFrame:
    training_is_parquet = training_path.endswith(".parquet")
    eval_is_parquet = eval_path.endswith(".parquet")
    if training_is_parquet != eval_is_parquet:
        raise ValueError("--eval-data must use the same file format as the training file")
    eval_df = _load_dataframe(eval_path)
    eval_columns = eval_df.columns.tolist()
    if eval_columns != training_columns:
        raise ValueError(
            "--eval-data columns must exactly match the training file columns and order: "
            f"training={training_columns} eval={eval_columns}"
        )
    return eval_df


def _iter_dataframe_batches(
    df: pd.DataFrame,
    batch_size: int,
    label_col_map: dict[str, str] | None = None,
) -> Iterator[dict[str, Any]]:
    if label_col_map is None:
        label_col_map = {}
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        actual_bs = len(batch_df)
        label_cols = set(label_col_map.values())
        feature_columns = {
            col: batch_df[col].tolist() for col in batch_df.columns if col not in label_cols
        }
        labels = {
            label_col: batch_df[label_col].tolist()
            for label_col in label_cols
            if label_col in batch_df.columns
        }
        yield {"features": feature_columns, "labels": labels, "_batch_size": actual_bs}


def _train_epoch_single(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dag: FeatureDag,
    df: pd.DataFrame,
    batch_size: int,
    label_col_map: dict[str, str] | None = None,
    output_kinds: dict[str, str] | None = None,
    prefetch_batches: int = 0,
    artifacts: TrainingArtifactManager | None = None,
    checkpoint_state: PeriodicCheckpointState | None = None,
    global_step: int = 0,
    skip_batches: int = 0,
    best_score: float = 0.0,
    stale_epochs: int = 0,
    best_epoch: int = 0,
    checkpoint_interval_steps: int = 0,
    checkpoint_interval_seconds: float = 0.0,
    epoch: int = 0,
    output_contract: NormalizedOutputContract | None = None,
) -> tuple[float, int]:
    if label_col_map is None:
        label_col_map = {}
    if output_kinds is None:
        output_kinds = {}
    if checkpoint_state is None:
        checkpoint_state = PeriodicCheckpointState()
    model.train()
    objective_engine = ObjectiveEngine(output_contract) if output_contract is not None else None
    total_loss = 0.0
    n_batches = 0
    batches = iter_preprocessed_batches(
        dag,
        _iter_dataframe_batches(df, batch_size, label_col_map),
        prefetch_batches=prefetch_batches,
    )
    skipped = 0
    batch_offset = skip_batches
    for batch in batches:
        if skipped < skip_batches:
            skipped += 1
            continue
        actual_bs = int(batch.get("_batch_size", 0))
        feature_tensors = batch["features"]
        batch_labels = batch["labels"]
        batch_values = batch.get(
            "_batch_values",
            {**feature_tensors, **batch_labels},
        )
        if objective_engine is not None:
            execution = model.forward_execution(feature_tensors)
            loss = objective_engine(execution, batch_values).total
        else:
            outputs = ensure_model_output(model(feature_tensors), output_kinds)
            loss = None
            for task_name, output in outputs.items():
                if output.kind == "probability":
                    continue
                label_col = label_col_map.get(task_name, task_name)
                if label_col in batch_labels:
                    labels = torch.tensor(batch_labels[label_col], dtype=torch.float32).view(
                        actual_bs, 1
                    )
                    task_loss = _single_task_loss(output.tensor, labels, output.kind)
                    loss = task_loss if loss is None else loss + task_loss
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        global_step += 1
        _maybe_save_periodic_checkpoint(
            artifacts,
            model,
            epoch=epoch,
            step=global_step,
            batch_in_epoch=batch_offset + n_batches,
            current_loss=loss.item(),
            state=checkpoint_state,
            interval_steps=checkpoint_interval_steps,
            interval_seconds=checkpoint_interval_seconds,
            best_score=best_score,
            stale_epochs=stale_epochs,
            best_epoch=best_epoch,
            optimizer=optimizer,
        )
        if n_batches == 0:
            available_labels = [c for c in sorted(set(label_col_map.values())) if c in df.columns]
            raise ValueError(
                "No supervised batches were processed. Check that the dataset exposes the "
                "configured label columns and that label_col_map matches the model outputs. "
                f"label_col_map={label_col_map} available_labels={available_labels}"
            )
    return total_loss / max(n_batches, 1), global_step


def _evaluate_single(
    model: torch.nn.Module,
    dag: FeatureDag,
    df: pd.DataFrame,
    batch_size: int,
    label_col_map: dict[str, str] | None = None,
    output_kinds: dict[str, str] | None = None,
    prefetch_batches: int = 0,
    output_contract: NormalizedOutputContract | None = None,
) -> dict[str, dict[str, float]]:
    if label_col_map is None:
        label_col_map = {}
    if output_kinds is None:
        output_kinds = {}
    model.eval()
    all_outputs: dict[str, list[np.ndarray]] = {}
    all_labels: dict[str, list[np.ndarray]] = {}
    metric_outputs: dict[str, list[np.ndarray]] = {}
    metric_labels: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        batches = iter_preprocessed_batches(
            dag,
            _iter_dataframe_batches(df, batch_size, label_col_map),
            prefetch_batches=prefetch_batches,
        )
        for batch in batches:
            actual_bs = int(batch.get("_batch_size", 0))
            feature_tensors = batch["features"]
            batch_labels = batch["labels"]
            batch_values = batch.get(
                "_batch_values",
                {**feature_tensors, **batch_labels},
            )
            outputs = (
                model.forward_execution(feature_tensors).nodes
                if output_contract is not None
                else ensure_model_output(model(feature_tensors), output_kinds)
            )
            if output_contract is not None:
                for metric in output_contract.metrics:
                    output = outputs.get(metric.source)
                    if output is None:
                        raise ValueError(
                            f"metric '{metric.name}' source '{metric.source}' is missing"
                        )
                    raw = batch_labels.get(metric.label, [])
                    labels = np.array(
                        [float(value) if value is not None else np.nan for value in raw],
                        dtype=np.float32,
                    )
                    valid = ~np.isnan(labels)
                    if metric.mask is not None:
                        valid &= evaluate_mask(
                            metric.mask,
                            batch_values,
                            len(labels),
                        )
                    if valid.any():
                        metric_outputs.setdefault(metric.name, []).append(
                            output.tensor.cpu().numpy().flatten()[valid]
                        )
                        metric_labels.setdefault(metric.name, []).append(labels[valid])
                continue
            for t, output in outputs.items():
                label_col = label_col_map.get(t, t)
                if label_col not in batch_labels:
                    continue
                all_outputs.setdefault(t, []).append(output.tensor.cpu().numpy().flatten())
                labels = np.asarray(batch_labels[label_col], dtype=np.float32)
                if len(labels) < actual_bs:
                    labels = np.pad(labels, (0, actual_bs - len(labels)), constant_values=0)
                all_labels.setdefault(t, []).append(labels)
    results: dict[str, dict[str, float]] = {}
    if output_contract is not None:
        for metric in output_contract.metrics:
            if not metric_outputs.get(metric.name):
                continue
            predictions = np.concatenate(metric_outputs[metric.name])
            labels = np.concatenate(metric_labels[metric.name])
            value = compute_metrics(
                labels,
                predictions,
                [metric.type],
                output_kind=output_contract.node_kinds[metric.source],
            )[metric.type]
            results.setdefault(metric.source, {})[metric.type] = value
        if not results:
            raise ValueError("No evaluation labels were available for output_contract metrics")
        return results
    for t, logits_list in all_outputs.items():
        logits_arr = np.concatenate(logits_list)
        labels_arr = (
            np.concatenate(all_labels.get(t, []))
            if all_labels.get(t)
            else np.zeros_like(logits_arr)
        )
        kind = output_kinds.get(t, "binary_logit")
        if kind == "binary_logit":
            probs = 1.0 / (1.0 + np.exp(-logits_arr))
            results[t] = {
                "logloss": float(
                    torch.nn.functional.binary_cross_entropy(
                        torch.tensor(probs, dtype=torch.float32),
                        torch.tensor(labels_arr, dtype=torch.float32),
                    ).item()
                ),
            }
        elif kind == "probability":
            results[t] = {
                "logloss": float(
                    torch.nn.functional.binary_cross_entropy(
                        torch.tensor(logits_arr, dtype=torch.float32),
                        torch.tensor(labels_arr, dtype=torch.float32),
                    ).item()
                ),
            }
        else:
            err = logits_arr - labels_arr
            results[t] = {"mse": float(np.mean(err * err)), "mae": float(np.mean(np.abs(err)))}
    if not results:
        raise ValueError(
            "No evaluation labels were available. Check the dataset and label_col_map. "
            f"label_col_map={label_col_map} columns={list(df.columns)}"
        )
    return results


def _predict_all(
    model: torch.nn.Module,
    dag: FeatureDag,
    df: pd.DataFrame,
    batch_size: int,
    prefetch_batches: int = 0,
) -> dict[str, np.ndarray]:
    model.eval()
    all_keys = None
    all_logits: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        batches = iter_preprocessed_batches(
            dag,
            _iter_dataframe_batches(df, batch_size),
            prefetch_batches=prefetch_batches,
        )
        for batch in batches:
            feature_tensors = batch["features"]
            outputs = ensure_model_output(model(feature_tensors))
            if all_keys is None:
                all_keys = outputs.names()
                all_logits = {k: [] for k in all_keys}
            for k in all_keys:
                all_logits[k].append(outputs.tensor(k).cpu().numpy().flatten())
    return {k: np.concatenate(v) for k, v in all_logits.items()}


def _single_task_loss(
    prediction: torch.Tensor, labels: torch.Tensor, output_kind: str
) -> torch.Tensor:
    if output_kind == "binary_logit":
        return F.binary_cross_entropy_with_logits(prediction, labels)
    if output_kind in {"regression", "score"}:
        return F.mse_loss(prediction, labels)
    raise ValueError(f"single-mode training does not support output kind: {output_kind}")


def _maybe_save_periodic_checkpoint(
    artifacts: TrainingArtifactManager | None,
    model: torch.nn.Module,
    *,
    epoch: int,
    step: int,
    batch_in_epoch: int,
    current_loss: float,
    state: PeriodicCheckpointState,
    interval_steps: int,
    interval_seconds: float,
    best_score: float,
    stale_epochs: int,
    best_epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    if artifacts is None:
        return
    interval_steps = max(int(interval_steps), 0)
    interval_seconds = max(float(interval_seconds), 0.0)
    if interval_steps <= 0 and interval_seconds <= 0:
        return

    now = time.perf_counter()
    steps_due = interval_steps > 0 and (step - state.last_step >= interval_steps)
    time_due = interval_seconds > 0 and (now - state.last_time >= interval_seconds)
    if not (steps_due or time_due):
        return

    resume_state = build_resume_state(
        checkpoint_kind="periodic",
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        next_epoch=epoch,
        global_step=step,
        best_score=best_score,
        stale_epochs=stale_epochs,
        best_epoch=best_epoch,
        periodic_checkpoint_seq=state.seq + 1,
        last_periodic_checkpoint_step=step,
        optimizer=optimizer,
    )
    state.seq += 1
    artifacts.save_checkpoint(
        model,
        epoch=epoch,
        step=step,
        score=current_loss,
        metric_name="train_loss",
        is_best=False,
        resume_state=resume_state,
        version=f"periodic-epoch-{epoch:04d}-step-{step:06d}-{state.seq:04d}",
    )
    state.last_step = step
    state.last_time = now


def _build_feature_dag(flow_config: FlowConfig, args: argparse.Namespace) -> FeatureDag:
    use_rust = bool(
        getattr(args, "use_rust_preprocess", False)
        or getattr(args, "require_rust_preprocess", False)
    )
    return FeatureDag(
        flow_config,
        debug_mode=getattr(args, "debug", 0) > 0,
        use_rust=use_rust,
        config_path=args.feature_config if use_rust else None,
        require_rust=bool(getattr(args, "require_rust_preprocess", False)),
    )


def _run_single(args: argparse.Namespace) -> None:
    feature_config = args.feature_config
    model_config = args.model_config
    data_paths = resolve_data_paths(args)

    if not model_config:
        raise SystemExit("--model-config is required for single mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.artifact_dir) / "logs",
        log_file=args.log_file,
        run_name=args.run_name,
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    flow_config = FlowConfig.from_yaml(feature_config)
    dag = _build_feature_dag(flow_config, args)
    feat_info = FeatureInfo(dag.sources, dag.node_defs, dag.feature_schemas, dag.execution_order)
    features = feat_info.feature_tuples()
    logger.info(
        "%d sources, %d ops → %d features",
        len(flow_config.sources),
        len(flow_config.operators),
        len(features),
    )

    built = build_model_for_dag(model_config, feat_info, device)
    label_col_map = dict(built.spec.get("label_col_map", {}))
    if built.spec.get("output_contract") is not None:
        for source in flow_config.label_sources:
            label_col_map.setdefault(source.name, source.name)
    from ..models.params import format_parameter_summary

    logger.info(
        "%s tasks=%s | %s",
        built.config.type,
        built.spec["task_names"],
        format_parameter_summary(built.model),
    )

    if args.resume_from and args.init_weights:
        raise SystemExit("--resume-from cannot be combined with --init-weights")
    resume_state: dict[str, Any] | None = None
    if args.resume_from:
        resume_state = _restore_model_and_state(built.model, args.resume_from)
    elif args.init_weights:
        load_init_weights(built.model, args.init_weights, device)

    cfg = train_config_from_args(args, export_path=args.publish_path or "")
    artifacts = TrainingArtifactManager.from_config(
        cfg.artifacts,
        model_name=args.model_name or built.config.type,
        model_type=built.config.type,
        artifact_root=args.artifact_dir,
        publish_path=args.publish_path or None,
        feature_config_path=feature_config,
        model_config_path=model_config,
    )
    artifacts.prepare(feature_config, model_config)
    cfg.export_path = str(artifacts.paths.published_weights_path)

    start_epoch = 1
    start_batch_in_epoch = 0
    global_step = 0
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    periodic_state = PeriodicCheckpointState()

    if resume_state is not None:
        global_step = int(resume_state.get("global_step", 0))
        best_score = float(resume_state.get("best_score", best_score))
        best_epoch = int(resume_state.get("best_epoch", best_epoch))
        stale_epochs = int(resume_state.get("stale_epochs", stale_epochs))
        start_epoch = int(resume_state.get("next_epoch", resume_state.get("epoch", 1)))
        start_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
        periodic_state.seq = int(resume_state.get("periodic_checkpoint_seq", 0))
        periodic_state.last_step = int(
            resume_state.get("last_periodic_checkpoint_step", global_step)
        )

    logger.info("[Data files] %s", describe_data_paths(data_paths))
    df = _load_dataframes(data_paths)
    if "ctr" in df.columns:
        df["ctr"] = df["ctr"].astype("Int64")
    if "cvr" in df.columns:
        df["cvr"] = df["cvr"].astype("Int64")
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype("Int64")

    if args.eval_data:
        train_df = df.sample(frac=1.0, random_state=42)
        test_df = _load_eval_dataframe(data_paths[0], df.columns.tolist(), args.eval_data)
        if "ctr" in test_df.columns:
            test_df["ctr"] = test_df["ctr"].astype("Int64")
        if "cvr" in test_df.columns:
            test_df["cvr"] = test_df["cvr"].astype("Int64")
        if "user_id" in test_df.columns:
            test_df["user_id"] = test_df["user_id"].astype("Int64")
    else:
        df_shuffled = df.sample(frac=1.0, random_state=42)
        n_train = int(len(df_shuffled) * 0.8)
        train_df = df_shuffled.iloc[:n_train]
        test_df = df_shuffled.iloc[n_train:]
    logger.info("[Data] train=%d test=%d", len(train_df), len(test_df))
    logger.info(
        "[Data detail] rows=%d batch_size=%d train_batches~%d eval_batches~%d labels=%s",
        len(df),
        args.batch_size,
        max(1, (len(train_df) + args.batch_size - 1) // max(args.batch_size, 1)),
        max(1, (len(test_df) + args.batch_size - 1) // max(args.batch_size, 1)),
        label_col_map,
    )

    model = built.model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume_state is not None:
        optimizer_state = resume_state.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = _train_epoch_single(
            model,
            optimizer,
            dag,
            train_df,
            args.batch_size,
            label_col_map,
            built.spec.get("output_kinds", {}),
            cfg.prefetch_batches,
            artifacts,
            periodic_state,
            global_step=global_step,
            skip_batches=start_batch_in_epoch if epoch == start_epoch else 0,
            best_score=best_score,
            stale_epochs=stale_epochs,
            best_epoch=best_epoch,
            checkpoint_interval_steps=cfg.checkpoint_interval_steps,
            checkpoint_interval_seconds=cfg.checkpoint_interval_seconds,
            epoch=epoch,
            output_contract=built.spec.get("output_contract"),
        )
        metrics = _evaluate_single(
            model,
            dag,
            test_df,
            args.batch_size,
            label_col_map,
            built.spec.get("output_kinds", {}),
            cfg.prefetch_batches,
            output_contract=built.spec.get("output_contract"),
        )
        score = min((v["logloss"] for v in metrics.values()), default=best_score)
        is_best = score < best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if artifacts is not None:
            artifacts.save_checkpoint(
                model,
                epoch=epoch,
                step=global_step,
                score=score,
                metric_name="logloss",
                is_best=is_best,
                resume_state=build_resume_state(
                    checkpoint_kind="epoch",
                    epoch=epoch,
                    batch_in_epoch=0,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                    best_score=best_score,
                    stale_epochs=stale_epochs,
                    best_epoch=best_epoch,
                    periodic_checkpoint_seq=periodic_state.seq,
                    last_periodic_checkpoint_step=periodic_state.last_step,
                    optimizer=optimizer,
                ),
            )
        logger.info("epoch %d/%d loss=%.6f", epoch, args.epochs, train_loss)

    if not np.isfinite(best_score):
        best_score = 0.0

    test_df.to_csv(
        artifacts.paths.published_weights_path.with_name(
            artifacts.paths.published_weights_path.stem + "_test.csv"
        )
    )
    preds = _predict_all(model, dag, test_df, args.batch_size, cfg.prefetch_batches)
    preds_rows = (
        {"label_ctr": test_df["ctr"].to_numpy().astype(np.float32)}
        if "ctr" in test_df.columns
        else {}
    )
    for key, values in preds.items():
        preds_rows[f"logit_{key}"] = values
    pd.DataFrame(preds_rows).to_csv(
        artifacts.paths.published_weights_path.with_name(
            artifacts.paths.published_weights_path.stem + "_py_preds.csv"
        )
    )
    artifacts.finalize(
        model=model,
        model_type=built.config.type,
        tasks=built.spec["task_names"],
        label_col_map=built.spec["label_col_map"],
        metrics={"best_score": best_score},
        repo_root=args.repo_root,
        task_specs=built.spec.get("tasks"),
        published_version=artifacts.best.version if artifacts.best is not None else None,
        best_score=best_score,
        published_source=artifacts.paths.best_alias_path if artifacts.best is not None else None,
    )


def _run_demo(args: argparse.Namespace) -> None:
    data_paths = resolve_data_paths(args)
    if not args.model_config:
        raise SystemExit("--model-config is required for demo mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.artifact_dir) / "logs",
        log_file=args.log_file,
        run_name=args.run_name,
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    fc = FlowConfig.from_yaml(args.feature_config)
    dag = _build_feature_dag(fc, args)
    feat_info = FeatureInfo(dag.sources, dag.node_defs, dag.feature_schemas, dag.execution_order)
    features = feat_info.feature_tuples()
    logger.info(
        "%d sources, %d ops → %d features", len(fc.sources), len(fc.operators), len(features)
    )

    built = build_model_for_dag(args.model_config, feat_info, device)
    from ..models.params import format_parameter_summary

    logger.info(
        "%s tasks=%s | %s",
        built.config.type,
        built.spec["task_names"],
        format_parameter_summary(built.model),
    )

    if args.resume_from and args.init_weights:
        raise SystemExit("--resume-from cannot be combined with --init-weights")
    if args.init_weights:
        load_init_weights(built.model, args.init_weights, device)

    cfg = train_config_from_args(args, export_path=args.publish_path or "")
    artifacts = TrainingArtifactManager.from_config(
        cfg.artifacts,
        model_name=args.model_name or built.config.type,
        model_type=built.config.type,
        artifact_root=args.artifact_dir,
        publish_path=args.publish_path or None,
        feature_config_path=args.feature_config,
        model_config_path=args.model_config,
    )
    artifacts.prepare(args.feature_config, args.model_config)
    cfg.export_path = str(artifacts.paths.published_weights_path)
    logger.info("feature config exported to %s", artifacts.paths.feature_config_path)
    logger.info("model config exported to %s", artifacts.paths.model_config_path)
    logger.info("[Data files] %s", describe_data_paths(data_paths))
    if args.eval_data:
        validate_matching_text_format(
            data_paths[0],
            args.eval_data,
            has_header=not args.no_header,
            sep=args.separator,
        )
        logger.info("[Validation file] %s", args.eval_data)
    label_col_map = built.spec.get("label_col_map", {})

    trainer = Trainer(
        built.model,
        TrainingPreprocessor(dag),
        built.spec["task_names"],
        label_col_map,
        device,
        cfg,
        model_type=built.config.type,
        flow_config=fc,
        data_paths=data_paths,
        eval_data_path=args.eval_data or None,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
        read_chunk_rows=args.read_chunk_rows,
        fast_no_na=args.fast_no_na,
        memory_map=args.memory_map,
        task_specs=built.spec.get("tasks"),
        output_kinds=built.spec.get("output_kinds", {}),
        output_contract=built.spec.get("output_contract"),
        task_metrics=built.spec.get("task_metrics"),
        artifact_manager=artifacts,
        repo_root=args.repo_root,
    )
    if args.resume_from:
        trainer.resume_from_checkpoint(args.resume_from)
    best = trainer.fit()
    logger.info("best metric=%.4f", best)


def _run_all(args: argparse.Namespace) -> None:
    data_paths = resolve_data_paths(args)

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.artifact_dir) / "logs",
        log_file=args.log_file,
        run_name=args.run_name,
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    flow_config = FlowConfig.from_yaml(args.feature_config)
    dag = _build_feature_dag(flow_config, args)
    feat_info = FeatureInfo(dag.sources, dag.node_defs, dag.feature_schemas, dag.execution_order)
    features = feat_info.feature_tuples()
    logger.info("%d features, %d ops", len(features), len(flow_config.operators))

    logger.info("[Data files] %s", describe_data_paths(data_paths))
    df = _load_dataframes(data_paths)
    for c in ["user_id", "ctr", "cvr"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    if args.eval_data:
        train_df = df.sample(frac=1.0, random_state=42)
        test_df = _load_eval_dataframe(data_paths[0], df.columns.tolist(), args.eval_data)
        for column in ["user_id", "ctr", "cvr"]:
            if column in test_df.columns:
                test_df[column] = test_df[column].astype("Int64")
    else:
        df_shuffled = df.sample(frac=1.0, random_state=42)
        n_train = int(len(df_shuffled) * 0.8)
        train_df = df_shuffled.iloc[:n_train]
        test_df = df_shuffled.iloc[n_train:]
    logger.info("[Data] train=%d test=%d", len(train_df), len(test_df))
    logger.info(
        "[Data detail] rows=%d batch_size=%d train_batches~%d eval_batches~%d",
        len(df),
        args.batch_size,
        max(1, (len(train_df) + args.batch_size - 1) // max(args.batch_size, 1)),
        max(1, (len(test_df) + args.batch_size - 1) // max(args.batch_size, 1)),
    )

    if args.resume_from and args.init_weights:
        raise SystemExit("--resume-from cannot be combined with --init-weights")

    model_specs = [
        ("demo_gdcn_esmm", args.model_config_demo_gdcn_esmm),
        ("demo_unimixer", args.model_config_demo_unimixer),
    ]
    if args.models == "all":
        selected = model_specs
    else:
        wanted = set(args.models.split(","))
        selected = [item for item in model_specs if item[0] in wanted]

    results = []
    for model_type, model_config_path in selected:
        if not model_config_path or not Path(model_config_path).exists():
            logger.info("[Skip] config not found: %s", model_config_path)
            continue
        built = build_model_for_dag(model_config_path, feat_info, device)
        model = built.model
        spec = built.spec
        label_col_map = spec.get("label_col_map", {})
        resume_state: dict[str, Any] | None = None
        if args.resume_from:
            resume_state = _restore_model_and_state(model, args.resume_from)
        elif args.init_weights:
            load_init_weights(model, args.init_weights, device)

        cfg = train_config_from_args(args, export_path=args.publish_path or "")
        model_name = f"{args.model_name}-{model_type}" if args.model_name else model_type
        artifacts = TrainingArtifactManager.from_config(
            cfg.artifacts,
            model_name=model_name,
            model_type=model_type,
            artifact_root=args.artifact_dir,
            publish_path=None,
            feature_config_path=args.feature_config,
            model_config_path=model_config_path,
        )
        artifacts.prepare(args.feature_config, model_config_path)
        cfg.export_path = str(artifacts.paths.published_weights_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_score = float("inf")
        best_epoch = 0
        stale_epochs = 0
        global_step = 0
        periodic_state = PeriodicCheckpointState()
        start_epoch = 1
        start_batch_in_epoch = 0
        if resume_state is not None:
            optimizer_state = resume_state.get("optimizer_state")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            global_step = int(resume_state.get("global_step", 0))
            best_score = float(resume_state.get("best_score", best_score))
            best_epoch = int(resume_state.get("best_epoch", best_epoch))
            stale_epochs = int(resume_state.get("stale_epochs", stale_epochs))
            start_epoch = int(resume_state.get("next_epoch", resume_state.get("epoch", 1)))
            start_batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
            periodic_state.seq = int(resume_state.get("periodic_checkpoint_seq", 0))
            periodic_state.last_step = int(
                resume_state.get("last_periodic_checkpoint_step", global_step)
            )

        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, global_step = _train_epoch_single(
                model,
                optimizer,
                dag,
                train_df,
                args.batch_size,
                label_col_map,
                spec.get("output_kinds", {}),
                cfg.prefetch_batches,
                artifacts,
                periodic_state,
                global_step=global_step,
                skip_batches=start_batch_in_epoch if epoch == start_epoch else 0,
                best_score=best_score,
                stale_epochs=stale_epochs,
                best_epoch=best_epoch,
                checkpoint_interval_steps=cfg.checkpoint_interval_steps,
                checkpoint_interval_seconds=cfg.checkpoint_interval_seconds,
                epoch=epoch,
            )
            metrics = _evaluate_single(
                model,
                dag,
                test_df,
                args.batch_size,
                label_col_map,
                spec.get("output_kinds", {}),
                cfg.prefetch_batches,
            )
            score = min(
                (m.get("logloss", best_score) for m in metrics.values()), default=best_score
            )
            is_best = score < best_score
            if is_best:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            artifacts.save_checkpoint(
                model,
                epoch=epoch,
                step=global_step,
                score=score,
                metric_name="logloss",
                is_best=is_best,
                resume_state=build_resume_state(
                    checkpoint_kind="epoch",
                    epoch=epoch,
                    batch_in_epoch=0,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                    best_score=best_score,
                    stale_epochs=stale_epochs,
                    best_epoch=best_epoch,
                    periodic_checkpoint_seq=periodic_state.seq,
                    last_periodic_checkpoint_step=periodic_state.last_step,
                    optimizer=optimizer,
                ),
            )
            logger.info("[%s] epoch %d/%d loss=%.6f", model_type, epoch, args.epochs, train_loss)

        if not np.isfinite(best_score):
            best_score = 0.0

        prefix = artifacts.paths.published_weights_path.with_suffix("")
        test_df.to_csv(prefix.with_name(prefix.name + "_test.csv"))
        preds = _predict_all(model, dag, test_df, args.batch_size, cfg.prefetch_batches)
        preds_rows = {"label_ctr": test_df["ctr"].to_numpy().astype(np.float32)}
        if "cvr" in test_df.columns:
            preds_rows["label_cvr"] = test_df["cvr"].to_numpy().astype(np.float32)
        for k, v in preds.items():
            preds_rows[f"logit_{k}"] = v
        pd.DataFrame(preds_rows).to_csv(prefix.with_name(prefix.name + "_py_preds.csv"))
        artifacts.finalize(
            model=model,
            model_type=model_type,
            tasks=spec["task_names"],
            label_col_map=label_col_map,
            metrics={"best_score": best_score},
            repo_root=args.repo_root,
            task_specs=spec.get("tasks"),
            published_version=artifacts.best.version if artifacts.best is not None else None,
            best_score=best_score,
            published_source=artifacts.paths.best_alias_path
            if artifacts.best is not None
            else None,
        )
        results.append(
            {
                "model_type": model_type,
                "best_score": best_score,
                "params": sum(p.numel() for p in model.parameters()),
            }
        )

    for r in results:
        logger.info(
            "%s best_score=%.4f params=%s", r["model_type"], r["best_score"], f"{r['params']:,}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified scale-rec training entry")
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="train a single model on a CSV/Parquet dataset")
    single.add_argument(
        "--feature-config", default=str(SHARED_EXAMPLES_DIR / "feature_config_demo.yaml")
    )
    single.add_argument("--model-config", required=True)
    add_data_range_args(single, data_required=False)
    single.add_argument(
        "--artifact-dir", "--export-dir", dest="artifact_dir", default=str(DEMO_ARTIFACT_DIR)
    )
    single.add_argument("--publish-path", "--export-path", dest="publish_path")
    single.add_argument("--debug", type=int, default=0)
    add_training_args(single, lr=0.001, batch_size=64)
    add_artifact_args(single)
    add_runtime_args(single)

    demo = sub.add_parser("demo", help="train demo-main-sort with TSV input")
    demo.add_argument(
        "--feature-config", default=str(SHARED_EXAMPLES_DIR / "feature_config_demo.yaml")
    )
    demo.add_argument("--model-config", required=True)
    add_data_range_args(demo, data_required=False)
    demo.add_argument(
        "--artifact-dir", "--export-dir", dest="artifact_dir", default=str(DEMO_ARTIFACT_DIR)
    )
    demo.add_argument("--publish-path", "--export-path", dest="publish_path")
    demo.add_argument("--no-header", action="store_true")
    demo.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    demo.add_argument("--separator", default="\t")
    demo.add_argument(
        "--read-chunk-rows",
        type=int,
        default=0,
        help="pandas read_csv chunk rows; 0 uses the batch-size based default",
    )
    demo.add_argument(
        "--fast-no-na",
        action="store_true",
        help="disable pandas NA detection for faster reads when defaults can be handled later",
    )
    demo.add_argument(
        "--memory-map",
        action="store_true",
        help="enable pandas memory_map for local uncompressed files",
    )
    add_training_args(demo, lr=0.005, batch_size=64)
    add_artifact_args(demo)
    add_runtime_args(demo)

    all_ = sub.add_parser("all", help="train multiple models on one dataset")
    all_.add_argument(
        "--feature-config", default=str(SHARED_EXAMPLES_DIR / "feature_config_demo.yaml")
    )
    add_data_range_args(all_, data_required=False)
    all_.add_argument(
        "--artifact-dir", "--export-dir", dest="artifact_dir", default=str(DEMO_ARTIFACT_DIR)
    )
    all_.add_argument("--publish-path", "--export-path", dest="publish_path")
    all_.add_argument("--models", default="all")
    all_.add_argument(
        "--model-config-demo-gdcn-esmm",
        default=str(MODEL_EXAMPLES_DIR / "gdcn_esmm.yaml"),
    )
    all_.add_argument(
        "--model-config-demo-unimixer",
        default=str(MODEL_EXAMPLES_DIR / "unimixer.yaml"),
    )
    all_.add_argument("--debug", type=int, default=0)
    add_training_args(all_, lr=0.005, batch_size=64)
    add_artifact_args(all_)
    add_runtime_args(all_)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.repo_root = REPO_ROOT

    if args.mode == "single":
        _run_single(args)
    elif args.mode == "demo":
        _run_demo(args)
    elif args.mode == "all":
        _run_all(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
