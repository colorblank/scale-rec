from __future__ import annotations

import argparse
import logging

import pytest
import torch
from safetensors.torch import save_file

from train.app.cli import (
    add_data_range_args,
    add_runtime_args,
    add_training_args,
    configure_logging,
    load_init_weights,
    train_config_from_args,
)


def test_train_config_from_args_uses_nested_config_fields(tmp_path):
    parser = argparse.ArgumentParser()
    add_training_args(parser, lr=0.1, batch_size=32)
    args = parser.parse_args(
        [
            "--optim",
            "sgd",
            "--batch-size",
            "128",
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

    assert cfg.batch_size == 128
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


def test_training_args_include_init_weights_default():
    parser = argparse.ArgumentParser()
    add_training_args(parser, lr=0.1, batch_size=32)

    args = parser.parse_args([])

    assert args.init_weights == ""
    assert args.resume_from == ""
    assert args.checkpoint_interval_steps is None
    assert args.checkpoint_interval_seconds is None


def test_data_range_args_are_optional_with_glob_support():
    parser = argparse.ArgumentParser()
    add_data_range_args(parser, data_required=False)

    args = parser.parse_args(
        [
            "--data-glob",
            "data/user_*.txt",
            "--start-date",
            "20260325",
            "--end-date",
            "20260331",
        ]
    )

    assert args.data is None
    assert args.eval_data == ""
    assert args.data_glob == "data/user_*.txt"
    assert args.start_date == "20260325"
    assert args.end_date == "20260331"


def test_demo_parser_includes_pandas_streaming_options():
    from train.app.main import build_parser

    args = build_parser().parse_args(
        [
            "demo",
            "--model-config",
            "examples/models/gdcn_esmm.yaml",
            "--run-name",
            "demo_train",
            "--data",
            "train.tsv",
            "--read-chunk-rows",
            "65536",
            "--fast-no-na",
            "--memory-map",
        ]
    )

    assert args.read_chunk_rows == 65536
    assert args.fast_no_na is True
    assert args.memory_map is True


def test_demo_parser_accepts_independent_eval_data():
    from train.app.main import build_parser

    args = build_parser().parse_args(
        [
            "demo",
            "--model-config",
            "examples/models/gdcn_esmm.yaml",
            "--run-name",
            "demo_eval",
            "--data",
            "train.tsv",
            "--eval-data",
            "eval.tsv",
        ]
    )

    assert args.eval_data == "eval.tsv"
    assert args.run_name == "demo_eval"


def test_training_subcommands_require_run_name():
    from train.app.main import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "demo",
                "--model-config",
                "examples/models/gdcn_esmm.yaml",
                "--data",
                "train.tsv",
            ]
        )


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


def test_load_init_weights_restores_parameters_strictly(tmp_path):
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    with torch.no_grad():
        source.weight.fill_(3.0)
        source.bias.fill_(1.0)
        target.weight.zero_()
        target.bias.zero_()
    weights = tmp_path / "model.safetensors"
    save_file(source.state_dict(), str(weights))

    load_init_weights(target, weights, torch.device("cpu"))

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_load_init_weights_reports_shape_mismatch(tmp_path):
    target = torch.nn.Linear(2, 1)
    weights = tmp_path / "bad.safetensors"
    save_file({"weight": torch.ones(3, 1), "bias": torch.ones(1)}, str(weights))

    with pytest.raises(RuntimeError, match="failed to load --init-weights"):
        load_init_weights(target, weights, torch.device("cpu"))


def _flush_root_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()
