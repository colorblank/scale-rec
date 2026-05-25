from __future__ import annotations

"""训练全流程配置。"""
from dataclasses import dataclass, field
from typing import Any

from .eval.evaluator import EvalConfig


@dataclass
class OptimConfig:
    """优化器与学习率配置。"""

    name: str = "adamw"  # adamw | adam | sgd
    lr: float = 0.005
    weight_decay: float = 1e-4
    momentum: float = 0.0
    emb_lr: float | None = None  # embedding 独立学习率, None=与 lr 相同
    emb_weight_decay: float | None = None  # embedding 独立 wd


@dataclass
class LRScheduleConfig:
    """学习率调度配置。"""

    warmup_steps: int = 200
    min_lr_ratio: float = 0.01


@dataclass(init=False)
class TrainConfig:
    """训练全流程配置 — 单一致信源，无写死值。"""

    epochs: int = 30
    batch_size: int = 64
    export_path: str = ""

    # 子配置
    optim: OptimConfig = field(default_factory=OptimConfig)
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # 数据
    eval_samples: int = 2000
    eval_interval: int = 50
    log_interval: int = 10

    # 损失
    loss_weighting: str = "static"  # static | equal | uncertainty
    task_weights: dict[str, float] | None = None

    # 正则化
    grad_max_norm: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 5

    # TensorBoard
    tb_dir: str = ""
    tb_grad_interval: int = 100

    def __init__(
        self,
        epochs: int = 30,
        batch_size: int = 64,
        export_path: str = "",
        optim: OptimConfig | dict[str, Any] | None = None,
        lr_schedule: LRScheduleConfig | dict[str, Any] | None = None,
        eval: EvalConfig | dict[str, Any] | None = None,
        eval_samples: int = 2000,
        eval_interval: int = 50,
        log_interval: int = 10,
        loss_weighting: str = "static",
        task_weights: dict[str, float] | None = None,
        grad_max_norm: float = 1.0,
        ema_decay: float = 0.999,
        early_stopping_patience: int = 5,
        tb_dir: str = "",
        tb_grad_interval: int = 100,
        **legacy: Any,
    ):
        optim_data = self._as_dict(optim)
        for key in ("lr", "weight_decay", "momentum", "emb_lr", "emb_weight_decay"):
            if key in legacy:
                optim_data[key] = legacy.pop(key)
        if "embedding_weight_decay" in legacy:
            optim_data["emb_weight_decay"] = legacy.pop("embedding_weight_decay")

        lr_schedule_data = self._as_dict(lr_schedule)
        for key in ("warmup_steps", "min_lr_ratio"):
            if key in legacy:
                lr_schedule_data[key] = legacy.pop(key)

        if "ema_enabled" in legacy:
            ema_enabled = legacy.pop("ema_enabled")
            if not ema_enabled:
                ema_decay = 0.0

        if legacy:
            unknown = ", ".join(sorted(legacy))
            raise TypeError(f"Unknown TrainConfig arguments: {unknown}")

        self.epochs = epochs
        self.batch_size = batch_size
        self.export_path = export_path
        self.optim = OptimConfig(**optim_data)
        self.lr_schedule = LRScheduleConfig(**lr_schedule_data)
        self.eval = eval if isinstance(eval, EvalConfig) else EvalConfig(**(eval or {}))
        self.eval_samples = eval_samples
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.loss_weighting = loss_weighting
        self.task_weights = task_weights
        self.grad_max_norm = grad_max_norm
        self.ema_decay = ema_decay
        self.early_stopping_patience = early_stopping_patience
        self.tb_dir = tb_dir
        self.tb_grad_interval = tb_grad_interval

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value.copy()
        return vars(value).copy()

    @property
    def warmup_steps(self) -> int:
        return self.lr_schedule.warmup_steps

    @property
    def min_lr_ratio(self) -> float:
        return self.lr_schedule.min_lr_ratio

    @property
    def lr(self) -> float:
        return self.optim.lr

    @property
    def weight_decay(self) -> float:
        return self.optim.weight_decay

    @property
    def embedding_weight_decay(self) -> float:
        return self.optim.emb_weight_decay or 0.0

    @property
    def ema_enabled(self) -> bool:
        return self.ema_decay > 0
