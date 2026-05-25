"""纯 LR 单任务 CTR 预估训练脚本。"""

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
from train.trainer import Trainer  # noqa: E402

from .paths import DEMO_ARTIFACT_DIR, DISCOVER_FEATURE_CONFIG, MODEL_CONFIGS, REPO_ROOT  # noqa: E402

logger = logging.getLogger("train")


def main():
    p = argparse.ArgumentParser(description="纯 LR 单任务 CTR 训练")
    p.add_argument("--data", required=True)
    p.add_argument("--feature-config", default=str(DISCOVER_FEATURE_CONFIG))
    p.add_argument("--model-config", default=str(MODEL_CONFIGS["lr"]))
    p.add_argument("--no-header", action="store_true")
    add_training_args(p, lr=0.01, batch_size=128)
    p.set_defaults(
        epochs=10, eval_samples=1000, eval_interval=200, warmup_steps=100, eval_metrics="auc,gauc"
    )
    add_runtime_args(p)
    args = p.parse_args()

    configure_logging(args.log_level)
    device = resolve_device(args.device)
    logger.info("device: %s", device)

    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info("%d features, %d ops", len(features), len(fc.operators))

    built = build_model_for_dag(args.model_config, dag, device)
    logger.info(
        "LR params=%s task=%s",
        f"{built.param_count:,}",
        built.spec["task_names"],
    )
    bundle = prepare_export_bundle(
        export_path=DEMO_ARTIFACT_DIR / "lr_ctr.safetensors",
        export_dir=DEMO_ARTIFACT_DIR,
        model_type=built.config.type,
        feature_config_path=args.feature_config,
        model_config_path=args.model_config,
        copy_configs=False,
    )

    cfg = train_config_from_args(args, export_path=bundle.export_path)
    trainer = Trainer(
        built.model,
        dag,
        built.spec["task_names"],
        built.spec["label_col_map"],
        device,
        cfg,
        data_path=args.data,
        flow_config=fc,
        has_header=not args.no_header,
        task_specs=built.spec.get("tasks"),
    )
    best = trainer.fit()
    logger.info("best=%.4f", best)

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
