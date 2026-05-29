from __future__ import annotations

"""统一训练入口。

一个文件覆盖三种场景：
1. `single`：单模型训练，适合 LR / DeepFM / ESMM / UniMixer / GDCN+ESMM
2. `discover`：discover-main-sort 训练，使用单文件 TSV
3. `all`：同一数据集上批量训练多个模型
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cli import (
    add_runtime_args,
    add_training_args,
    add_artifact_args,
    build_model_for_dag,
    configure_logging,
    resolve_device,
    train_config_from_args,
)
from .artifacts import TrainingArtifactManager
from ..core.config import FlowConfig
from ..core.dag import FeatureDag
from ..training.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEMO_ARTIFACT_DIR = REPO_ROOT / "python" / "artifacts" / "demo"

NULL_MARKERS: set[str] = {"NULL", "\\N", "null", "None", ""}

logger = logging.getLogger("train")


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


def _train_epoch_single(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dag: FeatureDag,
    df: pd.DataFrame,
    batch_size: int,
    label_col_map: dict[str, str] | None = None,
) -> float:
    if label_col_map is None:
        label_col_map = {}
    model.train()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size]
        actual_bs = len(batch_df)
        feature_tensors = dag.preprocess_batch(batch_df.to_dict("records"))
        outputs = model(feature_tensors)
        loss = None
        for task_name, logits in outputs.items():
            label_col = label_col_map.get(task_name, task_name)
            if label_col in batch_df.columns:
                labels = torch.tensor(batch_df[label_col].to_numpy(), dtype=torch.float32).view(
                    actual_bs, 1
                )
                task_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = task_loss if loss is None else loss + task_loss
        if loss is None:
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _evaluate_single(
    model: torch.nn.Module,
    dag: FeatureDag,
    df: pd.DataFrame,
    batch_size: int,
    label_col_map: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    if label_col_map is None:
        label_col_map = {}
    model.eval()
    all_outputs: dict[str, list[np.ndarray]] = {}
    all_labels: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]
            actual_bs = len(batch_df)
            feature_tensors = dag.preprocess_batch(batch_df.to_dict("records"))
            outputs = model(feature_tensors)
            for t, logits in outputs.items():
                label_col = label_col_map.get(t, t)
                if label_col not in batch_df.columns:
                    continue
                all_outputs.setdefault(t, []).append(logits.cpu().numpy().flatten())
                labels = batch_df[label_col].to_numpy().astype(np.float32)
                if len(labels) < actual_bs:
                    labels = np.pad(labels, (0, actual_bs - len(labels)), constant_values=0)
                all_labels.setdefault(t, []).append(labels)
    results: dict[str, dict[str, float]] = {}
    for t, logits_list in all_outputs.items():
        logits_arr = np.concatenate(logits_list)
        labels_arr = (
            np.concatenate(all_labels.get(t, []))
            if all_labels.get(t)
            else np.zeros_like(logits_arr)
        )
        probs = 1.0 / (1.0 + np.exp(-logits_arr))
        # Keep the metrics minimal here; single mode is mostly for smoke training.
        results[t] = {
            "logloss": float(
                torch.nn.functional.binary_cross_entropy(
                    torch.tensor(probs, dtype=torch.float32),
                    torch.tensor(labels_arr, dtype=torch.float32),
                ).item()
            ),
        }
    return results


def _predict_all(
    model: torch.nn.Module, dag: FeatureDag, df: pd.DataFrame, batch_size: int
) -> dict[str, np.ndarray]:
    model.eval()
    all_keys = None
    all_logits: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_df = df.iloc[start : start + batch_size]
            feature_tensors = dag.preprocess_batch(batch_df.to_dict("records"))
            outputs = model(feature_tensors)
            if all_keys is None:
                all_keys = list(outputs.keys())
                all_logits = {k: [] for k in all_keys}
            for k in all_keys:
                all_logits[k].append(outputs[k].cpu().numpy().flatten())
    return {k: np.concatenate(v) for k, v in all_logits.items()}


def _run_single(args: argparse.Namespace) -> None:
    feature_config = args.feature_config
    model_config = args.model_config
    data = args.data

    if not data:
        raise SystemExit("--data is required for single mode")
    if not model_config:
        raise SystemExit("--model-config is required for single mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.artifact_dir) / "logs",
        log_file=args.log_file,
        run_name="single_train",
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    flow_config = FlowConfig.from_yaml(feature_config)
    dag = FeatureDag(flow_config, debug_mode=args.debug > 0)
    features = dag.feature_tuples()
    logger.info(
        "%d sources, %d ops → %d features",
        len(flow_config.sources),
        len(flow_config.operators),
        len(features),
    )

    built = build_model_for_dag(model_config, dag, device)
    logger.info(
        "%s tasks=%s params=%s",
        built.config.type,
        built.spec["task_names"],
        f"{built.param_count:,}",
    )

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

    df = _load_dataframe(data)
    if "ctr" in df.columns:
        df["ctr"] = df["ctr"].astype("Int64")
    if "cvr" in df.columns:
        df["cvr"] = df["cvr"].astype("Int64")
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype("Int64")

    df_shuffled = df.sample(frac=1.0, random_state=42)
    n_train = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.iloc[:n_train]
    test_df = df_shuffled.iloc[n_train:]
    logger.info("[Data] train=%d test=%d", len(train_df), len(test_df))

    model = built.model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch_single(
            model, optimizer, dag, train_df, args.batch_size, built.spec.get("label_col_map", {})
        )
        metrics = _evaluate_single(
            model, dag, test_df, args.batch_size, built.spec.get("label_col_map", {})
        )
        best_score = min(
            best_score, min((v["logloss"] for v in metrics.values()), default=best_score)
        )
        logger.info("epoch %d/%d loss=%.6f", epoch, args.epochs, train_loss)

    if not np.isfinite(best_score):
        best_score = 0.0

    test_df.to_csv(
        artifacts.paths.published_weights_path.with_name(
            artifacts.paths.published_weights_path.stem + "_test.csv"
        )
    )
    preds = _predict_all(model, dag, test_df, args.batch_size)
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
        published_version=artifacts.best.version if artifacts.best is not None else None,
        best_score=best_score,
        published_source=artifacts.paths.best_alias_path if artifacts.best is not None else None,
    )


def _run_discover(args: argparse.Namespace) -> None:
    if not args.data:
        raise SystemExit("--data is required for discover mode")
    if not args.model_config:
        raise SystemExit("--model-config is required for discover mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.artifact_dir) / "logs",
        log_file=args.log_file,
        run_name="discover_train",
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info(
        "%d sources, %d ops → %d features", len(fc.sources), len(fc.operators), len(features)
    )

    built = build_model_for_dag(args.model_config, dag, device)
    logger.info(
        "%s tasks=%s params=%s",
        built.config.type,
        built.spec["task_names"],
        f"{built.param_count:,}",
    )

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

    trainer = Trainer(
        built.model,
        dag,
        built.spec["task_names"],
        built.spec["label_col_map"],
        device,
        cfg,
        model_type=built.config.type,
        data_path=args.data,
        flow_config=fc,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
        task_specs=built.spec.get("tasks"),
        artifact_manager=artifacts,
        repo_root=args.repo_root,
    )
    best = trainer.fit()
    logger.info("best AUC=%.4f", best)


def _run_all(args: argparse.Namespace) -> None:
    if not args.data:
        raise SystemExit("--data is required for all mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.artifact_dir) / "logs",
        log_file=args.log_file,
        run_name="all_train",
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    flow_config = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(flow_config, debug_mode=args.debug > 0)
    features = dag.feature_tuples()
    logger.info("%d features, %d ops", len(features), len(flow_config.operators))

    df = _load_dataframe(args.data)
    for c in ["user_id", "ctr", "cvr"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    df_shuffled = df.sample(frac=1.0, random_state=42)
    n_train = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.iloc[:n_train]
    test_df = df_shuffled.iloc[n_train:]
    logger.info("[Data] train=%d test=%d", len(train_df), len(test_df))

    model_specs = [
        ("discover_gdcn_esmm", args.model_config_discover_gdcn_esmm),
        ("discover_unimixer", args.model_config_discover_unimixer),
    ]
    if args.models == "all":
        selected = model_specs
    else:
        wanted = set(args.models.split(","))
        selected = [item for item in model_specs if item[0] in wanted]

    results = []
    for model_type, model_config_path in selected:
        if not model_config_path or not os.path.exists(model_config_path):
            logger.info("[Skip] config not found: %s", model_config_path)
            continue
        built = build_model_for_dag(model_config_path, dag, device)
        model = built.model
        spec = built.spec
        label_col_map = spec.get("label_col_map", {})

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
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch_single(
                model, optimizer, dag, train_df, args.batch_size, label_col_map
            )
            metrics = _evaluate_single(model, dag, test_df, args.batch_size, label_col_map)
            score = min(
                (m.get("logloss", best_score) for m in metrics.values()), default=best_score
            )
            is_best = score < best_score
            if is_best:
                best_score = score
            artifacts.save_checkpoint(
                model,
                epoch=epoch,
                step=epoch,
                score=score,
                metric_name="logloss",
                is_best=is_best,
            )
            logger.info("[%s] epoch %d/%d loss=%.6f", model_type, epoch, args.epochs, train_loss)

        if not np.isfinite(best_score):
            best_score = 0.0

        prefix = artifacts.paths.published_weights_path.with_suffix("")
        test_df.to_csv(prefix.with_name(prefix.name + "_test.csv"))
        preds = _predict_all(model, dag, test_df, args.batch_size)
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
        "--feature-config", default=str(EXAMPLES_DIR / "feature_config_discover.yaml")
    )
    single.add_argument("--model-config", required=True)
    single.add_argument("--data", required=True)
    single.add_argument(
        "--artifact-dir", "--export-dir", dest="artifact_dir", default=str(DEMO_ARTIFACT_DIR)
    )
    single.add_argument("--publish-path", "--export-path", dest="publish_path")
    single.add_argument("--debug", type=int, default=0)
    add_training_args(single, lr=0.001, batch_size=64)
    add_artifact_args(single)
    add_runtime_args(single)

    discover = sub.add_parser("discover", help="train discover-main-sort with TSV input")
    discover.add_argument(
        "--feature-config", default=str(EXAMPLES_DIR / "feature_config_discover.yaml")
    )
    discover.add_argument("--model-config", required=True)
    discover.add_argument("--data", required=True)
    discover.add_argument(
        "--artifact-dir", "--export-dir", dest="artifact_dir", default=str(DEMO_ARTIFACT_DIR)
    )
    discover.add_argument("--publish-path", "--export-path", dest="publish_path")
    discover.add_argument("--no-header", action="store_true")
    discover.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    discover.add_argument("--separator", default="\t")
    add_training_args(discover, lr=0.005, batch_size=64)
    add_artifact_args(discover)
    add_runtime_args(discover)

    all_ = sub.add_parser("all", help="train multiple models on one dataset")
    all_.add_argument(
        "--feature-config", default=str(EXAMPLES_DIR / "feature_config_discover.yaml")
    )
    all_.add_argument("--data", required=True)
    all_.add_argument(
        "--artifact-dir", "--export-dir", dest="artifact_dir", default=str(DEMO_ARTIFACT_DIR)
    )
    all_.add_argument("--publish-path", "--export-path", dest="publish_path")
    all_.add_argument("--models", default="all")
    all_.add_argument(
        "--model-config-discover-gdcn-esmm", default=str(EXAMPLES_DIR / "model_gdcn_esmm.yaml")
    )
    all_.add_argument(
        "--model-config-discover-unimixer",
        default=str(EXAMPLES_DIR / "model_discover_unimixer.yaml"),
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
    elif args.mode == "discover":
        _run_discover(args)
    elif args.mode == "all":
        _run_all(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
