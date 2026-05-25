from __future__ import annotations

import argparse

from train.cli import add_training_args, train_config_from_args


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
