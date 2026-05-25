"""discover-main-sort 训练脚本。

pandas chunk read 流式读取单文件 TSV，头部取验证集，每 epoch 重读训练。
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from train.config import FlowConfig  # noqa: E402
from train.config_train import TrainConfig  # noqa: E402
from train.dag import FeatureDag  # noqa: E402
from train.data import build_item_index, stream_join  # noqa: E402
from train.manifest import write_model_manifest  # noqa: E402
from train.models import ModelConfig, get_output_spec  # noqa: E402
from train.trainer import Trainer  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = DEMO_DIR.parent.parent

DEFAULT_FEATURE_CONFIG = str(_PROJ_ROOT / "examples" / "feature_config_discover.yaml")
DEFAULT_MODEL_CONFIG = str(DEMO_DIR / "model_discover_esmm.yaml")

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
    p.add_argument("--skip-missing-item", action="store_true")
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=200, help="LR warmup steps 数")
    p.add_argument("--min-lr-ratio", type=float, default=0.01, help="cosine decay 最终 lr 比例")
    p.add_argument("--grad-max-norm", type=float, default=1.0, help="梯度裁剪阈值 (0=禁用)")
    p.add_argument("--early-stopping", type=int, default=5, help="early stopping patience (0=禁用)")
    p.add_argument("--no-ema", action="store_true", help="禁用 EMA")
    p.add_argument("--ema-decay", type=float, default=0.999, help="EMA 衰减率")
    p.add_argument("--tb-dir", default="", help="TensorBoard 日志目录（空=禁用）")
    p.add_argument(
        "--loss-weighting",
        default="static",
        choices=["equal", "static", "uncertainty"],
        help="多任务损失加权模式",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    if bool(args.data) == bool(args.user_data):
        p.error("--data and --user-data are mutually exclusive; provide exactly one")
    if args.user_data and not args.item_files:
        p.error("--item-files is required with --user-data")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
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

    # Model
    mc = ModelConfig.from_yaml(args.model_config)

    # UniMixer 需要预构建 FeatureTokenizer
    tokenizer = None
    if mc.type == "unimixer":
        from train.models.unimixer.tokenizer import FeatureTokenizer

        p = mc.params
        td = p.get("token_dim", 64)
        nt = p.get("num_tokens", 8)
        tokenizer = FeatureTokenizer(features, td, nt, pooling_map=dag.feature_pooling())
        logger.info("tokenizer: %d features → %d tokens × %dd", len(features), nt, td)

    model = mc.build(
        features,
        tokenizer=tokenizer,
        pooling_map=dag.feature_pooling(),
        total_dim=dag.feature_total_dim(),
    )

    # UniMixer: 包装 state_dict 对齐 Rust vb.pp("unimixer") 命名
    if mc.type == "unimixer":
        import torch.nn as nn

        blocks = model.blocks
        task_towers = model.task_towers
        final_norm = model.final_norm
        tokenizer_mod = model.tokenizer
        wrapper = nn.Module()
        wrapper.add_module("tokenizer", tokenizer_mod)
        inner = nn.Module()
        inner.add_module("blocks", blocks)
        inner.add_module("task_towers", task_towers)
        if final_norm is not None:
            inner.add_module("final_norm", final_norm)
        wrapper.add_module("unimixer", inner)
        _raw = model

        def _forward(self, x_inputs, temperature=None):
            return _raw(x_inputs, temperature)

        import types

        wrapper.forward = types.MethodType(_forward, wrapper)
        model = wrapper

    model = model.to(device)
    spec = get_output_spec(mc.type, model, mc.params)
    n = sum(p.numel() for p in model.parameters())
    logger.info("%s  tasks=%s  params=%s", mc.type, spec["task_names"], f"{n:,}")

    # Export path: {model_type}_{YYYYMMDD_HHMMSS}.safetensors
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_dir = Path(args.export_path).parent if args.export_path else DEMO_DIR / "temp"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = str(export_dir / f"{mc.type}_{ts}.safetensors")

    # 导出配置副本，manifest 以副本为发布契约。
    feature_config_copy = export_dir / f"feature_config_{ts}.yaml"
    model_config_copy = export_dir / f"model_config_{ts}.yaml"
    shutil.copy(args.feature_config, feature_config_copy)
    shutil.copy(args.model_config, model_config_copy)
    logger.info("feature config exported to %s", feature_config_copy)
    logger.info("model config exported to %s", model_config_copy)

    # Train
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_samples=args.eval_samples,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        export_path=export_path,
        optim={"lr": args.lr, "weight_decay": args.weight_decay},
        lr_schedule={"warmup_steps": args.warmup_steps, "min_lr_ratio": args.min_lr_ratio},
        grad_max_norm=args.grad_max_norm,
        early_stopping_patience=args.early_stopping,
        ema_decay=0.0 if args.no_ema else args.ema_decay,
        tb_dir=args.tb_dir,
        loss_weighting=args.loss_weighting,
    )
    trainer = Trainer(
        model,
        dag,
        spec["task_names"],
        spec["label_col_map"],
        device,
        cfg,
        data_path=data_path,
        flow_config=fc,
        has_header=not args.no_header,
        sep=args.separator,
        null_markers=set(args.null_markers),
        batch_factory=batch_factory,
    )
    best = trainer.fit()
    logger.info("best AUC=%.4f", best)

    manifest_path = export_dir / f"{Path(export_path).stem}.manifest.yaml"
    write_model_manifest(
        manifest_path=manifest_path,
        model_id=Path(export_path).stem,
        model_version=ts,
        model_type=mc.type,
        weights_path=export_path,
        feature_config_path=feature_config_copy,
        model_config_path=model_config_copy,
        tasks=spec["task_names"],
        label_col_map=spec["label_col_map"],
        metrics={"best_auc": float(best)},
        repo_root=_PROJ_ROOT,
    )
    logger.info("manifest exported to %s", manifest_path)


if __name__ == "__main__":
    main()
