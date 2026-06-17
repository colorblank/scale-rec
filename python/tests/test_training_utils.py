from __future__ import annotations

import threading
import time

import pytest
import torch

from train.core.config import EvalConfig, FlowConfig, LRScheduleConfig, OptimConfig, TrainConfig
from train.core.dag import FeatureDag
from train.core.preprocessor import TrainingPreprocessor
from train.core.task import TaskSpec, parse_task_specs
from train.training.eval.evaluator import Evaluator
from train.training.loss.multi_task import MultiTaskLoss
from train.training.optim.scheduler import LRScheduler
from train.training.quality import summarize_feature_quality
from train.training.trainer import Trainer, iter_preprocessed_batches


def _single_label_flow(label: str = "click") -> FlowConfig:
    return FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": "", "role": "feature"},
                {"name": label, "dtype": "int", "default_val": "0", "role": "label"},
            ],
            "operators": [],
        }
    )


def _dummy_preprocessor(flow_config: FlowConfig) -> TrainingPreprocessor:
    return TrainingPreprocessor(FeatureDag(flow_config))


def test_train_config_uses_structured_subconfigs():
    cfg = TrainConfig(
        optim=OptimConfig(lr=0.01, weight_decay=0.02, emb_weight_decay=0.0),
        lr_schedule=LRScheduleConfig(warmup_steps=7, min_lr_ratio=0.2),
        ema_decay=0.0,
    )

    assert cfg.optim.lr == 0.01
    assert cfg.optim.weight_decay == 0.02
    assert cfg.optim.emb_weight_decay == 0.0
    assert cfg.lr_schedule.warmup_steps == 7
    assert cfg.lr_schedule.min_lr_ratio == 0.2
    assert cfg.prefetch_batches == 2
    assert cfg.checkpoint_interval_steps == 0
    assert cfg.checkpoint_interval_seconds == 0.0
    assert not cfg.ema_enabled


def test_train_config_rejects_legacy_kwargs():
    with pytest.raises(TypeError):
        TrainConfig(lr=0.01)  # type: ignore[call-arg]


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


def test_eval_config_rejects_unknown_monitor_mode():
    with pytest.raises(ValueError, match="monitor_mode"):
        EvalConfig(monitor_mode="largest")


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


def test_task_spec_drives_loss_label_weight_and_mask():
    specs = parse_task_specs(
        [
            {
                "name": "click",
                "label": "is_click",
                "loss": "bce",
                "weight": 2.0,
                "mask": "is_click >= 0",
            }
        ]
    )
    loss_fn = MultiTaskLoss(["click"], {"click": "wrong"}, task_specs=specs)
    outputs = {"click": torch.zeros(3, 1)}
    labels = {"is_click": [0, -1, 1]}

    loss = loss_fn(outputs, labels)

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(1.3862944))
    assert loss_fn.task_weights_info() == {"click": 2.0}


def test_multi_task_loss_rejects_unknown_model_outputs():
    loss_fn = MultiTaskLoss(["click"], {"click": "click"})

    with pytest.raises(ValueError, match="not covered by task specs"):
        loss_fn({"click": torch.zeros(1, 1), "extra": torch.zeros(1, 1)}, {"click": [1]})


def test_multi_task_loss_allows_untrained_probability_relation_outputs():
    loss_fn = MultiTaskLoss(
        ["click"],
        {"click": "click"},
        output_kinds={"click": "binary_logit", "ctcvr": "probability"},
    )

    loss = loss_fn({"click": torch.zeros(1, 1), "ctcvr": torch.ones(1, 1)}, {"click": [1]})

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(0.6931472))


def test_multi_task_loss_rejects_probability_relation_as_training_task():
    specs = parse_task_specs(
        [
            {
                "name": "ctcvr",
                "label": "is_cvr",
                "loss": "bce",
                "output_kind": "probability",
            }
        ]
    )
    loss_fn = MultiTaskLoss(["ctcvr"], {"ctcvr": "is_cvr"}, task_specs=specs)

    with pytest.raises(ValueError, match="requires binary_logit"):
        loss_fn({"ctcvr": torch.ones(1, 1)}, {"is_cvr": [1]})


def test_multi_task_loss_supports_regression_outputs():
    specs = parse_task_specs(
        [
            {
                "name": "watch_time",
                "label": "watch_time",
                "loss": "mse",
                "metrics": ["mae", "mse"],
            }
        ]
    )
    loss_fn = MultiTaskLoss(
        ["watch_time"],
        {"watch_time": "watch_time"},
        task_specs=specs,
    )

    loss = loss_fn(
        {"watch_time": torch.tensor([[1.0], [3.0]])},
        {"watch_time": [2.0, 1.0]},
    )

    assert loss is not None
    assert torch.isclose(loss, torch.tensor(2.5))


def test_multi_task_loss_rejects_bce_on_regression_output():
    specs = parse_task_specs(
        [
            {
                "name": "click",
                "label": "click",
                "loss": "bce",
                "output_kind": "regression",
            }
        ]
    )
    loss_fn = MultiTaskLoss(["click"], {"click": "click"}, task_specs=specs)

    with pytest.raises(ValueError, match="requires binary_logit"):
        loss_fn({"click": torch.zeros(1, 1)}, {"click": [1]})


def test_multi_task_loss_rejects_missing_model_outputs():
    loss_fn = MultiTaskLoss(["click", "cvr"], {"click": "click", "cvr": "cvr"})

    with pytest.raises(ValueError, match="missing"):
        loss_fn({"click": torch.zeros(1, 1)}, {"click": [1], "cvr": [0]})


def test_trainer_monitor_score_uses_configured_task_metric_and_mode():
    flow_config = _single_label_flow()
    preproc = _dummy_preprocessor(flow_config)
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        preproc,
        task_names=["click"],
        label_map={"click": "click"},
        device=torch.device("cpu"),
        config=TrainConfig(
            eval={"metrics": ["auc", "logloss"], "monitor_task": "click", "monitor_metric": "auc"}
        ),
        task_specs=[TaskSpec(name="click", label="click", metrics=("auc", "logloss"))],
        data_path="unused",
        flow_config=flow_config,
    )

    assert trainer._monitor_score({"click": {"auc": 0.7, "logloss": 0.3}}) == 0.7
    assert trainer._is_better(0.8, 0.7)


def test_trainer_monitor_auto_minimizes_loss_metrics():
    flow_config = _single_label_flow()
    preproc = _dummy_preprocessor(flow_config)
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        preproc,
        task_names=["click"],
        label_map={"click": "click"},
        device=torch.device("cpu"),
        config=TrainConfig(eval={"metrics": ["logloss"], "monitor_metric": "logloss"}),
        task_specs=[TaskSpec(name="click", label="click", metrics=("logloss",))],
        data_path="unused",
        flow_config=flow_config,
    )

    assert trainer._monitor_score({"click": {"logloss": 0.3}}) == 0.3
    assert trainer._is_better(0.2, 0.3)
    assert not trainer._is_better(0.4, 0.3)


def test_trainer_iter_batches_reads_all_data_paths(tmp_path):
    first = tmp_path / "part_20260325.tsv"
    second = tmp_path / "part_20260326.tsv"
    first.write_text("user_id\tis_click\nu1\t1\n", encoding="utf-8")
    second.write_text("user_id\tis_click\nu2\t0\n", encoding="utf-8")
    flow_config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": "", "role": "feature"},
                {"name": "is_click", "dtype": "int", "default_val": "0", "role": "label"},
            ],
            "operators": [],
        }
    )
    preproc = _dummy_preprocessor(flow_config)
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        preproc,
        task_names=["click"],
        label_map={"click": "is_click"},
        device=torch.device("cpu"),
        config=TrainConfig(batch_size=1, eval={"metrics": ["auc"]}),
        task_specs=[TaskSpec(name="click", label="is_click", metrics=("auc",))],
        flow_config=flow_config,
        data_paths=[str(first), str(second)],
    )

    batches = list(trainer._iter_batches())

    assert [batch["features"]["user_id"] for batch in batches] == [["u1"], ["u2"]]
    assert [batch["labels"]["is_click"] for batch in batches] == [[1], [0]]


def test_trainer_prefetches_batches_in_order():
    class FakeDag:
        tracer = None

        def __init__(self) -> None:
            self.calls: list[str] = []

        def preprocess_batch(self, rows):
            self.calls.append(threading.current_thread().name)
            time.sleep(0.01)
            values = rows["user_id"]
            return {"user_id": torch.tensor(values, dtype=torch.long)}

    dag = FakeDag()
    batches = [
        {"features": {"user_id": [1]}, "labels": {"is_click": [1]}},
        {"features": {"user_id": [2]}, "labels": {"is_click": [0]}},
        {"features": {"user_id": [3]}, "labels": {"is_click": [1]}},
    ]

    prepared_batches = list(iter_preprocessed_batches(dag, iter(batches), prefetch_batches=2))

    assert [batch["labels"]["is_click"] for batch in prepared_batches] == [[1], [0], [1]]
    assert [batch["features"]["user_id"].tolist() for batch in prepared_batches] == [[1], [2], [3]]
    assert all(name != threading.current_thread().name for name in dag.calls)


def test_trainer_prefetch_surfaces_background_errors():
    class RaisingDag:
        tracer = None

        def preprocess_batch(self, rows):
            if rows["user_id"][0] == 2:
                raise RuntimeError("boom")
            return {"user_id": torch.tensor(rows["user_id"], dtype=torch.long)}

    batches = [
        {"features": {"user_id": [1]}, "labels": {"is_click": [1]}},
        {"features": {"user_id": [2]}, "labels": {"is_click": [0]}},
        {"features": {"user_id": [3]}, "labels": {"is_click": [1]}},
    ]

    with pytest.raises(RuntimeError, match="boom"):
        list(iter_preprocessed_batches(RaisingDag(), iter(batches), prefetch_batches=2))


def test_trainer_periodic_checkpoint_saves_by_step_interval():
    class FakeArtifacts:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def save_checkpoint(
            self,
            model,
            *,
            epoch,
            step,
            score,
            metric_name,
            is_best,
            resume_state=None,
            version=None,
        ):
            self.calls.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "score": score,
                    "metric_name": metric_name,
                    "is_best": is_best,
                    "resume_state": resume_state,
                    "version": version,
                }
            )

    flow_config = _single_label_flow()
    preproc = _dummy_preprocessor(flow_config)
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        preproc,
        task_names=["click"],
        label_map={"click": "click"},
        device=torch.device("cpu"),
        task_specs=[TaskSpec(name="click", label="click", metrics=("logloss",))],
        data_path="unused",
        flow_config=flow_config,
        config=TrainConfig(
            checkpoint_interval_steps=2,
            checkpoint_interval_seconds=0.0,
            eval={"metrics": ["logloss"], "monitor_metric": "logloss"},
        ),
    )
    trainer.artifacts = FakeArtifacts()
    trainer._global_step = 2
    trainer._last_periodic_checkpoint_step = 0

    trainer._maybe_save_periodic_checkpoint(
        epoch=1,
        batch_in_epoch=2,
        current_loss=0.123,
    )

    resume = trainer.artifacts.calls[0]["resume_state"]
    assert trainer.artifacts.calls[0]["epoch"] == 1
    assert trainer.artifacts.calls[0]["step"] == 2
    assert trainer.artifacts.calls[0]["score"] == 0.123
    assert trainer.artifacts.calls[0]["metric_name"] == "train_loss"
    assert trainer.artifacts.calls[0]["is_best"] is False
    assert trainer.artifacts.calls[0]["version"] == "periodic-epoch-0001-step-000002-0001"
    assert resume["schema_version"] == 1
    assert resume["checkpoint_kind"] == "periodic"
    assert resume["epoch"] == 1
    assert resume["batch_in_epoch"] == 2
    assert resume["next_epoch"] == 1
    assert resume["global_step"] == 2
    assert resume["best_score"] == float("inf")
    assert resume["stale_epochs"] == 0
    assert resume["best_epoch"] == 0
    assert resume["periodic_checkpoint_seq"] == 1
    assert resume["last_periodic_checkpoint_step"] == 2
    assert "optimizer_state" not in resume
    assert not resume["loss_fn_state"]


def test_trainer_periodic_checkpoint_saves_by_time_interval(monkeypatch):
    class FakeArtifacts:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def save_checkpoint(
            self,
            model,
            *,
            epoch,
            step,
            score,
            metric_name,
            is_best,
            resume_state=None,
            version=None,
        ):
            self.calls.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "score": score,
                    "metric_name": metric_name,
                    "is_best": is_best,
                    "resume_state": resume_state,
                    "version": version,
                }
            )

    flow_config = _single_label_flow()
    preproc = _dummy_preprocessor(flow_config)
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        preproc,
        task_names=["click"],
        label_map={"click": "click"},
        device=torch.device("cpu"),
        task_specs=[TaskSpec(name="click", label="click", metrics=("logloss",))],
        data_path="unused",
        flow_config=flow_config,
        config=TrainConfig(
            checkpoint_interval_steps=0,
            checkpoint_interval_seconds=1.0,
            eval={"metrics": ["logloss"], "monitor_metric": "logloss"},
        ),
    )
    trainer.artifacts = FakeArtifacts()
    trainer._global_step = 1
    trainer._last_periodic_checkpoint_step = 1
    trainer._last_periodic_checkpoint_time = 0.0
    monkeypatch.setattr("train.training.trainer.time.perf_counter", lambda: 1.5)

    trainer._maybe_save_periodic_checkpoint(
        epoch=1,
        batch_in_epoch=1,
        current_loss=0.456,
    )

    resume = trainer.artifacts.calls[0]["resume_state"]
    assert trainer.artifacts.calls[0]["epoch"] == 1
    assert trainer.artifacts.calls[0]["step"] == 1
    assert trainer.artifacts.calls[0]["score"] == 0.456
    assert trainer.artifacts.calls[0]["metric_name"] == "train_loss"
    assert trainer.artifacts.calls[0]["is_best"] is False
    assert trainer.artifacts.calls[0]["version"] == "periodic-epoch-0001-step-000001-0001"
    assert resume["schema_version"] == 1
    assert resume["checkpoint_kind"] == "periodic"
    assert resume["batch_in_epoch"] == 1
    assert resume["global_step"] == 1
    assert resume["periodic_checkpoint_seq"] == 1
    assert resume["last_periodic_checkpoint_step"] == 1
    assert "optimizer_state" not in resume
    assert not resume["loss_fn_state"]


def test_trainer_validation_comes_from_last_data_path(tmp_path):
    first = tmp_path / "part_20260325.tsv"
    second = tmp_path / "part_20260326.tsv"
    first.write_text("user_id\tis_click\nu1\t1\n", encoding="utf-8")
    second.write_text("user_id\tis_click\nu2\t0\nu3\t1\n", encoding="utf-8")
    flow_config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": "", "role": "feature"},
                {"name": "is_click", "dtype": "int", "default_val": "0", "role": "label"},
            ],
            "operators": [],
        }
    )
    trainer = Trainer(
        torch.nn.Linear(1, 1),
        TrainingPreprocessor(FeatureDag(flow_config)),
        task_names=["click"],
        label_map={"click": "is_click"},
        device=torch.device("cpu"),
        config=TrainConfig(batch_size=1, eval_samples=1, eval={"metrics": ["auc"]}),
        task_specs=[TaskSpec(name="click", label="is_click", metrics=("auc",))],
        flow_config=flow_config,
        data_paths=[str(first), str(second)],
    )

    trainer._collect_eval()
    train_batches = list(trainer._iter_batches())

    assert [batch["features"]["user_id"] for batch in trainer.eval_batches] == [["u2"]]
    assert [batch["features"]["user_id"] for batch in train_batches] == [["u1"], ["u3"]]


def test_feature_quality_report_tracks_missing_defaults_and_buckets():
    config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": ""},
                {"name": "age", "dtype": "float", "default_val": "0.0"},
            ],
            "operators": [
                {
                    "name": "user_hash",
                    "op_type": "FeatureHash",
                    "inputs": ["user_id"],
                    "outputs": ["user_id_idx"],
                    "params": {"vocab_size": 16, "num_hashes": 1},
                    "embed": {"vocab_size": 16, "embed_dim": 4},
                },
                {
                    "name": "age_bucket",
                    "op_type": "Bucketing",
                    "inputs": ["age"],
                    "outputs": ["age_bucket"],
                    "params": {"boundaries": [18.0, 30.0]},
                    "embed": {"vocab_size": 4, "embed_dim": 4},
                },
            ],
        }
    )
    dag = FeatureDag(config)
    report = summarize_feature_quality(
        dag.executor,
        dag.feat_info,
        [
            {
                "features": [
                    {"user_id": "u1", "age": 20.0},
                    {"age": 0.0},
                ],
                "labels": {},
            }
        ],
    )

    assert report.rows == 2
    assert report.sources["user_id"].missing_rate == 0.5
    assert report.sources["age"].default_rate == 0.5
    assert report.embeddables["user_id_idx"].unique_buckets >= 1
    assert report.embeddables["user_id_idx"].truncations == 0
    assert report.embeddables["age_bucket"].truncations == 0
    assert "user_id_idx" in report.hash_cache
    cache = report.hash_cache["user_id_idx"]
    assert cache.total == 2
    assert cache.cache_size == 2
    assert cache.hit_rate == 0.0
    metrics = report.to_metrics()
    assert metrics["feature_quality.source.user_id.missing_rate"] == 0.5
    assert "feature_quality.emb.age_bucket.bucket_utilization" in metrics
    assert metrics["feature_quality.hash_cache.user_id_idx.hit_rate"] == 0.0
    assert metrics["feature_quality.hash_cache.user_id_idx.cache_size"] == 2.0


def test_feature_quality_counts_sequence_padding_as_empty():
    config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "interest_keywords", "dtype": "string", "default_val": ""},
            ],
            "operators": [
                {
                    "name": "kw_parse",
                    "op_type": "StringParser",
                    "inputs": ["interest_keywords"],
                    "outputs": ["kw_tokens"],
                    "params": {
                        "sep1": "|",
                        "sep2": "#",
                        "key_index": 0,
                        "pad_len": 3,
                        "pad_val": "",
                    },
                },
                {
                    "name": "kw_hash",
                    "op_type": "FeatureHash",
                    "inputs": ["kw_tokens"],
                    "outputs": ["kw_ids"],
                    "params": {"vocab_size": 100, "num_hashes": 1},
                    "embed": {
                        "vocab_size": 100,
                        "embed_dim": 4,
                        "pooling": "mean",
                        "seq_len": 3,
                    },
                },
            ],
        }
    )
    dag = FeatureDag(config)

    report = summarize_feature_quality(
        dag.executor,
        dag.feat_info,
        [
            {
                "features": [
                    {"interest_keywords": "ai#0.8"},
                    {"interest_keywords": ""},
                ],
                "labels": {},
            }
        ],
    )

    stat = report.embeddables["kw_ids"]
    assert stat.empty_sequence_rate == 0.5
    assert stat.truncation_rate == 1.0
    assert stat.mean_length == 0.5
    assert stat.padding_rate == pytest.approx(5 / 6)
    assert report.to_metrics()["feature_quality.emb.kw_ids.truncation_rate"] == 1.0
    assert report.to_metrics()["feature_quality.emb.kw_ids.padding_rate"] == pytest.approx(5 / 6)
    assert "kw_ids" in report.hash_cache
    assert report.hash_cache["kw_ids"].total == 6
    assert report.hash_cache["kw_ids"].cache_size > 0


def test_parameter_counting_utility():
    from train.models.params import count_parameters, format_parameter_summary

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb_user = torch.nn.Embedding(10, 8)
            self.dense_layer = torch.nn.Linear(8, 4)
            self.frozen_param = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    model = DummyModel()
    counts = count_parameters(model)

    assert counts["emb_trainable"] == 80
    assert counts["dense_trainable"] == 36  # weight (4 * 8 = 32) + bias (4)
    assert counts["non_trainable"] == 2
    assert counts["total"] == 118

    summary = format_parameter_summary(model)
    assert "total=118" in summary
    assert "emb_trainable=80" in summary
    assert "dense_trainable=36" in summary
    assert "non_trainable=2" in summary
