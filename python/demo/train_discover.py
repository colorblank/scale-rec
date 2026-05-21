"""discover-main-sort 训练脚本。

pandas chunk read 流式读取单文件 TSV，头部取验证集，每 epoch 重读训练。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(DEMO_DIR))

DEFAULT_FEATURE_CONFIG = os.path.join(_PROJ_ROOT, "examples", "feature_config_discover.yaml")
DEFAULT_MODEL_CONFIG = os.path.join(DEMO_DIR, "model_discover_esmm.yaml")

NULL_MARKERS: set[str] = {"NULL", "\\N", "null", "None", ""}
logger = logging.getLogger("train")


def main() -> None:
    p = argparse.ArgumentParser(description="discover-main-sort 训练")
    p.add_argument("--data", required=True, help="训练数据 TSV 路径")
    p.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--export-path")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-header", action="store_true", help="TSV 文件无 header 行")
    p.add_argument("--null-markers", nargs="*", default=list(NULL_MARKERS))
    p.add_argument("--separator", default="\t")
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info("device: %s", device)

    # DAG
    fc = FlowConfig.from_yaml(args.feature_config)
    dag = FeatureDag(fc)
    features = dag.feature_tuples()
    logger.info("%d sources, %d ops → %d features", len(fc.sources), len(fc.operators), len(features))

    # Model
    mc = ModelConfig.from_yaml(args.model_config)
    spec = get_output_spec(mc.type, None)
    model = mc.build(features).to(device)
    n = sum(p.numel() for p in model.parameters())
    logger.info("%s  tasks=%s  params=%s", mc.type, spec["task_names"], f"{n:,}")

    # Train
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_samples=args.eval_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        export_path=args.export_path or os.path.join(DEMO_DIR, "temp", "model.safetensors"),
    )
    trainer = Trainer(
        model, dag, spec["task_names"], spec["label_col_map"],
        device, cfg,
        data_path=args.data,
        flow_config=fc,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
    )
    best = trainer.fit()
    logger.info("best AUC=%.4f", best)


if __name__ == "__main__":
    main()
