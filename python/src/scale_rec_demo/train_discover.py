"""discover-main-sort 训练脚本。

pandas chunk read 流式读取单文件 TSV，头部取验证集，每 epoch 重读训练。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.cli import (  # noqa: E402
    add_runtime_args,
    add_training_args,
    build_model_for_dag,
    configure_logging,
    prepare_export_bundle,
    resolve_device,
    train_config_from_args,
    write_training_manifest,
)
from train.config import FlowConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.data import build_item_index, stream_join  # noqa: E402
from train.trainer import Trainer  # noqa: E402

from .paths import DEMO_ARTIFACT_DIR, DISCOVER_FEATURE_CONFIG, MODEL_CONFIGS, REPO_ROOT  # noqa: E402

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


def main() -> None:
    p = argparse.ArgumentParser(description="discover-main-sort 训练")
    p.add_argument("--data", help="single training TSV path")
    p.add_argument("--user-data", help="user behavior TSV path for streaming join mode")
    p.add_argument("--item-files", help="comma-separated item TSV paths for streaming join mode")
    p.add_argument("--feature-config", default=str(DISCOVER_FEATURE_CONFIG))
    p.add_argument("--model-config", default=str(MODEL_CONFIGS["discover_esmm"]))
    p.add_argument("--export-path")
    p.add_argument("--no-header", action="store_true", help="TSV 文件无 header 行")
    p.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    p.add_argument("--separator", default="\t")
    p.add_argument("--skip-missing-item", action="store_true")
    add_training_args(p, lr=0.005, batch_size=64)
    add_runtime_args(p)
    args = p.parse_args()
    if bool(args.data) == bool(args.user_data):
        p.error("--data and --user-data are mutually exclusive; provide exactly one")
    if args.user_data and not args.item_files:
        p.error("--item-files is required with --user-data")

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    # DAG
    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info(
        "%d sources, %d ops → %d features", len(fc.sources), len(fc.operators), len(features)
    )

    data_path = args.data or args.user_data
    batch_factory = None
    if args.user_data:
        all_sources = [_source_to_dict(s) for s in fc.sources]
        item_sources = [s for s in all_sources if s.get("source") in {"Item", "ItemStats"}]
        item_index = build_item_index(
            [x for x in args.item_files.split(",") if x],
            item_sources,
            has_header=not args.no_header,
            separator=args.separator,
            null_markers=set(args.null_markers),
        )

        def batch_factory():
            return stream_join(
                args.user_data,
                item_index,
                all_sources,
                batch_size=args.batch_size,
                separator=args.separator,
                has_header=not args.no_header,
                null_markers=set(args.null_markers),
                skip_missing_item=args.skip_missing_item,
            )

    built = build_model_for_dag(args.model_config, dag, device)
    logger.info(
        "%s  tasks=%s  params=%s",
        built.config.type,
        built.spec["task_names"],
        f"{built.param_count:,}",
    )
    bundle = prepare_export_bundle(
        export_path=args.export_path,
        export_dir=DEMO_ARTIFACT_DIR,
        model_type=built.config.type,
        feature_config_path=args.feature_config,
        model_config_path=args.model_config,
        copy_configs=True,
    )
    logger.info("feature config exported to %s", bundle.feature_config_path)
    logger.info("model config exported to %s", bundle.model_config_path)

    # Train
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

    manifest_path = write_training_manifest(
        bundle=bundle,
        model_id=bundle.export_path.stem,
        model_type=built.config.type,
        spec=built.spec,
        best_score=best,
        repo_root=REPO_ROOT,
    )
    logger.info("manifest exported to %s", manifest_path)


if __name__ == "__main__":
    main()
