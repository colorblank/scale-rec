from __future__ import annotations

import torch

from train.config_train import TrainConfig
from train.eval.evaluator import EvalConfig, Evaluator
from train.loss.multi_task import MultiTaskLoss
from train.optim.scheduler import LRScheduler
from train.trainer import Trainer


def test_train_config_accepts_legacy_kwargs():
    cfg = TrainConfig(
        lr=0.01,
        weight_decay=0.02,
        embedding_weight_decay=0.0,
        warmup_steps=7,
        min_lr_ratio=0.2,
        ema_enabled=False,
    )

    assert cfg.optim.lr == 0.01
    assert cfg.optim.weight_decay == 0.02
    assert cfg.optim.emb_weight_decay == 0.0
    assert cfg.lr_schedule.warmup_steps == 7
    assert cfg.lr_schedule.min_lr_ratio == 0.2
    assert not cfg.ema_enabled


def test_scheduler_preserves_param_group_lr_ratios():
    p1 = torch.nn.Parameter(torch.ones(1))
    p2 = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p1], "lr": 0.001},
            {"params": [p2], "lr": 0.01},
        ]
    )
    scheduler = LRScheduler([optimizer], warmup_steps=10, total_steps=100, min_lr_ratio=0.1)

    scheduler.step(5)
    assert optimizer.param_groups[0]["lr"] == 0.0005
    assert optimizer.param_groups[1]["lr"] == 0.005


def test_evaluator_masks_logits_when_labels_are_missing():
    class DummyDag:
        def preprocess_batch(self, rows):
            return {"x": torch.arange(len(rows), dtype=torch.float32).view(-1, 1)}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))

        def forward(self, inputs):
            return {"click": inputs["x"] + self.bias}

    batches = [
        {
            "features": [{"user_id": "u1"}, {"user_id": "u1"}, {"user_id": "u2"}],
            "labels": {"click": [0, None, 1]},
        }
    ]
    evaluator = Evaluator(EvalConfig(metrics=["auc", "gauc"]))

    result = evaluator.evaluate(
        DummyModel(),
        DummyDag(),
        batches,
        ["click"],
        {"click": "click"},
        torch.device("cpu"),
    )

    assert result["click"]["auc"] == 1.0
    assert result["click"]["gauc"] == 0.5


def test_eval_config_accepts_comma_separated_metrics():
    cfg = EvalConfig(metrics="auc,gauc")

    assert cfg.metrics == ["auc", "gauc"]


def test_equal_loss_ignores_static_task_weights():
    loss_fn = MultiTaskLoss(
        ["click"],
        {"click": "click"},
        mode="equal",
        task_weights={"click": 100.0},
    )
    outputs = {"click": torch.zeros(2, 1)}
    labels = {"click": [0, 1]}

    loss = loss_fn(outputs, labels)

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.6931472))
    assert loss_fn.task_weights_info() == {"click": 1.0}


def test_trainer_monitor_score_falls_back_to_configured_metric():
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        dag=None,
        task_names=["click"],
        label_map={"click": "click"},
        device=torch.device("cpu"),
        config=TrainConfig(eval={"metrics": ["gauc"]}),
        data_path="unused",
        flow_config=None,
    )

    assert trainer._monitor_score({"click": {"gauc": 0.7}}) == 0.7
