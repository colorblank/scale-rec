from __future__ import annotations

"""统一训练入口。

一个文件覆盖三种场景：
1. `single`：单模型训练，适合 LR / DeepFM / ESMM / UniMixer / GDCN+ESMM
2. `discover`：discover-main-sort 训练，支持单文件 TSV 或流式 join
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
    build_model_for_dag,
    configure_logging,
    prepare_export_bundle,
    resolve_device,
    train_config_from_args,
    write_training_manifest,
)
from ..core.config import FlowConfig
from ..core.dag import FeatureDag
from .data import build_item_index, stream_join
from .export import export_to_safetensors
from ..training.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEMO_ARTIFACT_DIR = REPO_ROOT / "python" / "artifacts" / "demo"

NULL_MARKERS: set[str] = {"NULL", "\\N", "null", "None", ""}

logger = logging.getLogger("train")


def _dtype_to_raw(dtype):
    if dtype.tag == "list":
        return {"list": {"dtype": _dtype_to_raw(dtype.inner), "length": dtype.length}}
    return dtype.tag


def _source_to_dict(source):
    return {
        "name": source.name,
        "source": source.source,
        "dtype": _dtype_to_raw(source.dtype),
        "default_val": source.default_val,
        "role": source.role,
        "column_index": source.column_index,
    }


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
        labels_arr = np.concatenate(all_labels.get(t, [])) if all_labels.get(t) else np.zeros_like(
            logits_arr
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


def _predict_all(model: torch.nn.Module, dag: FeatureDag, df: pd.DataFrame, batch_size: int) -> dict[str, np.ndarray]:
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


def _build_batch_factory(
    *,
    fc: FlowConfig,
    user_data: str,
    item_files: str,
    batch_size: int,
    separator: str,
    no_header: bool,
    null_markers: set[str],
    skip_missing_item: bool,
):
    all_sources = [_source_to_dict(s) for s in fc.sources]
    item_sources = [s for s in all_sources if s.get("source") in {"Item", "ItemStats"}]
    item_index = build_item_index(
        [x for x in item_files.split(",") if x],
        item_sources,
        has_header=not no_header,
        separator=separator,
        null_markers=null_markers,
    )

    def batch_factory():
        return stream_join(
            user_data,
            item_index,
            all_sources,
            batch_size=batch_size,
            separator=separator,
            has_header=not no_header,
            null_markers=null_markers,
            skip_missing_item=skip_missing_item,
        )

    return batch_factory


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
        log_dir=args.log_dir or Path(args.export_dir) / "logs",
        log_file=args.log_file,
        run_name="single_train",
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    flow_config = FlowConfig.from_yaml(feature_config)
    dag = FeatureDag(flow_config, debug_mode=args.debug > 0)
    features = dag.feature_tuples()
    logger.info("%d sources, %d ops → %d features", len(flow_config.sources), len(flow_config.operators), len(features))

    built = build_model_for_dag(model_config, dag, device)
    logger.info("%s tasks=%s params=%s", built.config.type, built.spec["task_names"], f"{built.param_count:,}")

    export_path = args.export_path or (Path(args.export_dir) / f"{built.config.type}.safetensors")
    bundle = prepare_export_bundle(
        export_path=export_path,
        export_dir=args.export_dir,
        model_type=built.config.type,
        feature_config_path=feature_config,
        model_config_path=model_config,
        copy_configs=True,
    )

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
        metrics = _evaluate_single(model, dag, test_df, args.batch_size, built.spec.get("label_col_map", {}))
        best_score = min(best_score, min((v["logloss"] for v in metrics.values()), default=best_score))
        logger.info("epoch %d/%d loss=%.6f", epoch, args.epochs, train_loss)

    if not np.isfinite(best_score):
        best_score = 0.0

    export_to_safetensors(model, bundle.export_path)
    test_df.to_csv(bundle.export_path.with_name(bundle.export_path.stem + "_test.csv"))
    preds = _predict_all(model, dag, test_df, args.batch_size)
    preds_rows = {"label_ctr": test_df["ctr"].to_numpy().astype(np.float32)} if "ctr" in test_df.columns else {}
    for key, values in preds.items():
        preds_rows[f"logit_{key}"] = values
    pd.DataFrame(preds_rows).to_csv(bundle.export_path.with_name(bundle.export_path.stem + "_py_preds.csv"))

    write_training_manifest(
        bundle=bundle,
        model_id=bundle.export_path.stem,
        model_type=built.config.type,
        spec=built.spec,
        best_score=best_score,
        extra_metrics={},
        repo_root=args.repo_root,
    )


def _run_discover(args: argparse.Namespace) -> None:
    if bool(args.data) == bool(args.user_data):
        raise SystemExit("--data and --user-data are mutually exclusive; provide exactly one")
    if args.user_data and not args.item_files:
        raise SystemExit("--item-files is required with --user-data")
    if not args.model_config:
        raise SystemExit("--model-config is required for discover mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.export_dir) / "logs",
        log_file=args.log_file,
        run_name="discover_train",
    )
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info("%d sources, %d ops → %d features", len(fc.sources), len(fc.operators), len(features))

    data_path = args.data or args.user_data
    batch_factory = None
    if args.user_data:
        batch_factory = _build_batch_factory(
            fc=fc,
            user_data=args.user_data,
            item_files=args.item_files,
            batch_size=args.batch_size,
            separator=args.separator,
            no_header=args.no_header,
            null_markers=set(args.null_markers),
            skip_missing_item=args.skip_missing_item,
        )

    built = build_model_for_dag(args.model_config, dag, device)
    logger.info("%s tasks=%s params=%s", built.config.type, built.spec["task_names"], f"{built.param_count:,}")

    bundle = prepare_export_bundle(
        export_path=args.export_path,
        export_dir=args.export_dir,
        model_type=built.config.type,
        feature_config_path=args.feature_config,
        model_config_path=args.model_config,
        copy_configs=True,
    )
    logger.info("feature config exported to %s", bundle.feature_config_path)
    logger.info("model config exported to %s", bundle.model_config_path)

    cfg = train_config_from_args(args, export_path=bundle.export_path)
    trainer = Trainer(
        built.model,
        dag,
        built.spec["task_names"],
        built.spec["label_col_map"],
        device,
        cfg,
        data_path=data_path,
        flow_config=fc,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
        batch_factory=batch_factory,
        task_specs=built.spec.get("tasks"),
    )
    best = trainer.fit()
    logger.info("best AUC=%.4f", best)

    write_training_manifest(
        bundle=bundle,
        model_id=bundle.export_path.stem,
        model_type=built.config.type,
        spec=built.spec,
        best_score=best,
        extra_metrics=trainer.feature_quality_metrics(),
        repo_root=args.repo_root,
    )


def _run_all(args: argparse.Namespace) -> None:
    if not args.data:
        raise SystemExit("--data is required for all mode")

    configure_logging(
        args.log_level,
        file_level=args.file_log_level,
        log_dir=args.log_dir or Path(args.export_dir) / "logs",
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
        ("lr", args.model_config_lr),
        ("deepfm", args.model_config_deepfm),
        ("mmoe", args.model_config_mmoe),
        ("esmm", args.model_config_esmm),
        ("unimixer", args.model_config_unimixer),
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

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_score = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = _train_epoch_single(
                model, optimizer, dag, train_df, args.batch_size, label_col_map
            )
            metrics = _evaluate_single(model, dag, test_df, args.batch_size, label_col_map)
            score = min((m.get("logloss", best_score) for m in metrics.values()), default=best_score)
            best_score = min(best_score, score)
            logger.info("[%s] epoch %d/%d loss=%.6f", model_type, epoch, args.epochs, train_loss)

        if not np.isfinite(best_score):
            best_score = 0.0

        prefix = Path(args.export_dir) / f"model_{model_type}"
        export_to_safetensors(model, prefix.with_suffix(".safetensors"))
        test_df.to_csv(prefix.with_name(prefix.name + "_test.csv"))
        preds = _predict_all(model, dag, test_df, args.batch_size)
        preds_rows = {"label_ctr": test_df["ctr"].to_numpy().astype(np.float32)}
        if "cvr" in test_df.columns:
            preds_rows["label_cvr"] = test_df["cvr"].to_numpy().astype(np.float32)
        for k, v in preds.items():
            preds_rows[f"logit_{k}"] = v
        pd.DataFrame(preds_rows).to_csv(prefix.with_name(prefix.name + "_py_preds.csv"))
        results.append(
            {
                "model_type": model_type,
                "best_score": best_score,
                "params": sum(p.numel() for p in model.parameters()),
            }
        )

    for r in results:
        logger.info("%s best_score=%.4f params=%s", r["model_type"], r["best_score"], f"{r['params']:,}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified scale-rec training entry")
    sub = parser.add_subparsers(dest="mode", required=True)

    single = sub.add_parser("single", help="train a single model on a CSV/Parquet dataset")
    single.add_argument("--feature-config", default=str(EXAMPLES_DIR / "feature_config_legacy.yaml"))
    single.add_argument("--model-config", required=True)
    single.add_argument("--data", required=True)
    single.add_argument("--export-dir", default=str(DEMO_ARTIFACT_DIR))
    single.add_argument("--export-path")
    single.add_argument("--debug", type=int, default=0)
    add_training_args(single, lr=0.001, batch_size=64)
    add_runtime_args(single)

    discover = sub.add_parser("discover", help="train discover-main-sort with TSV input")
    discover.add_argument("--feature-config", default=str(EXAMPLES_DIR / "feature_config_discover.yaml"))
    discover.add_argument("--model-config", required=True)
    discover.add_argument("--data")
    discover.add_argument("--user-data")
    discover.add_argument("--item-files")
    discover.add_argument("--export-dir", default=str(DEMO_ARTIFACT_DIR))
    discover.add_argument("--export-path")
    discover.add_argument("--no-header", action="store_true")
    discover.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    discover.add_argument("--separator", default="\t")
    discover.add_argument("--skip-missing-item", action="store_true")
    add_training_args(discover, lr=0.005, batch_size=64)
    add_runtime_args(discover)

    all_ = sub.add_parser("all", help="train multiple models on one dataset")
    all_.add_argument("--feature-config", default=str(EXAMPLES_DIR / "feature_config_legacy.yaml"))
    all_.add_argument("--data", required=True)
    all_.add_argument("--export-dir", default=str(DEMO_ARTIFACT_DIR))
    all_.add_argument("--models", default="all")
    all_.add_argument("--model-config-lr", default=str(EXAMPLES_DIR / "model_lr.yaml"))
    all_.add_argument("--model-config-deepfm", default=str(EXAMPLES_DIR / "model_deepfm.yaml"))
    all_.add_argument("--model-config-mmoe", default=str(EXAMPLES_DIR / "model_mmoe.yaml"))
    all_.add_argument("--model-config-esmm", default=str(EXAMPLES_DIR / "model_esmm.yaml"))
    all_.add_argument("--model-config-unimixer", default=str(EXAMPLES_DIR / "model_unimixer.yaml"))
    all_.add_argument("--debug", type=int, default=0)
    add_training_args(all_, lr=0.005, batch_size=64)
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
