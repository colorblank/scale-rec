from __future__ import annotations

import argparse
import logging

from train.cli import add_runtime_args, add_training_args, configure_logging, train_config_from_args


def test_train_config_from_args_uses_nested_config_fields(tmp_path):
    parser = argparse.ArgumentParser()
    add_training_args(parser, lr=0.1, batch_size=32)
    args = parser.parse_args(
        [
            "--optim",
            "sgd",
            "--lr",
            "0.2",
            "--weight-decay",
            "0.03",
            "--emb-lr",
            "0.04",
            "--emb-weight-decay",
            "0.005",
            "--eval-metrics",
            "auc,gauc",
            "--monitor-metric",
            "gauc",
            "--gauc-group-feature",
            "uid",
            "--no-ema",
        ]
    )

    cfg = train_config_from_args(args, export_path=tmp_path / "model.safetensors")

    assert cfg.batch_size == 32
    assert cfg.optim.name == "sgd"
    assert cfg.optim.lr == 0.2
    assert cfg.optim.weight_decay == 0.03
    assert cfg.optim.emb_lr == 0.04
    assert cfg.optim.emb_weight_decay == 0.005
    assert cfg.eval.metrics == ["auc", "gauc"]
    assert cfg.eval.monitor_metric == "gauc"
    assert cfg.eval.gauc_group_feature == "uid"
    assert cfg.ema_decay == 0.0


def test_runtime_args_include_file_logging_options():
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)

    args = parser.parse_args([])

    assert args.log_level == "INFO"
    assert args.file_log_level == "DEBUG"
    assert args.log_dir == ""
    assert args.log_file == ""


def test_configure_logging_writes_debug_file_logs(tmp_path):
    log_path = tmp_path / "train.log"
    configure_logging("WARNING", file_level="DEBUG", log_file=log_path)
    logger = logging.getLogger("train.test.logging")

    logger.debug("debug persisted")
    logger.info("info persisted")
    logger.warning("warning persisted")
    _flush_root_handlers()

    text = log_path.read_text()
    assert "debug persisted" in text
    assert "info persisted" in text
    assert "warning persisted" in text


def test_configure_logging_uses_log_dir_and_replaces_handlers(tmp_path):
    created = configure_logging("INFO", log_dir=tmp_path, run_name="demo train")
    assert created is not None
    assert created.parent == tmp_path
    assert created.name.startswith("demo_train_")

    configure_logging("INFO", file_level="INFO", log_file=created)
    logger = logging.getLogger("train.test.logging")
    logger.info("written once")
    _flush_root_handlers()

    text = created.read_text()
    assert text.count("written once") == 1


def _flush_root_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
