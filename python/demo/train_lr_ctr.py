"""纯 LR 单任务 CTR 预估训练脚本。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402
from train.trainer import TrainConfig, Trainer  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = DEMO_DIR.parent.parent

logger = logging.getLogger("train")


def main():
    p = argparse.ArgumentParser(description="纯 LR 单任务 CTR 训练")
    p.add_argument("--data", required=True)
    p.add_argument(
        "--feature-config", default=str(_PROJ_ROOT / "examples" / "feature_config_discover.yaml")
    )
    p.add_argument("--model-config", default=str(DEMO_DIR / "model_lr_ctr.yaml"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--no-header", action="store_true")
    p.add_argument("--eval-samples", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--device", default="auto")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(
        args.device
        if args.device != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("device: %s", device)

    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info("%d features, %d ops", len(features), len(fc.operators))

    mc = ModelConfig.from_yaml(args.model_config)
    model = mc.build(
        features, pooling_map=dag.feature_pooling(), total_dim=dag.feature_total_dim()
    ).to(device)
    spec = get_output_spec(mc.type, model)
    logger.info(
        "LR params=%s  task=%s",
        f"{sum(p.numel() for p in model.parameters()):,}",
        spec["task_names"],
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_samples=args.eval_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        warmup_steps=args.warmup_steps,
        export_path=str(DEMO_DIR / "temp" / "lr_ctr.safetensors"),
    )
    trainer = Trainer(
        model,
        dag,
        spec["task_names"],
        spec["label_col_map"],
        device,
        cfg,
        data_path=args.data,
        flow_config=fc,
        has_header=not args.no_header,
    )
    best = trainer.fit()
    logger.info("best AUC=%.4f", best)


if __name__ == "__main__":
    main()
