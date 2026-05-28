from __future__ import annotations

"""训练与推理配置的单一入口。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn
import yaml

from ..layers.embedding import FeatureTuple
from .task import TaskSpec, parse_task_specs


class Role:
    """列角色常量 — mirrors src/feats/config.rs Role enum."""

    FEATURE = "feature"
    LABEL = "label"
    DISCARD = "discard"


@dataclass
class DType:
    tag: str
    inner: Optional["DType"] = None
    max_len: int | None = None
    values: list[str] | None = None
    default: str | None = None
    oov: str | None = None

    @property
    def length(self) -> int:
        return self.max_len or 0

    @classmethod
    def from_dict(cls, raw: str | dict) -> "DType":
        if isinstance(raw, str):
            return cls(tag=raw)
        if isinstance(raw, dict) and "list" in raw:
            spec = raw["list"]
            inner_raw = spec.get("item_dtype", spec.get("dtype"))
            max_len = spec.get("max_len", spec.get("length"))
            if inner_raw is None:
                raise ValueError(f"Invalid list DType: {raw}")
            if max_len is None:
                raise ValueError(f"list dtype requires max_len: {raw}")
            return cls(tag="list", inner=cls.from_dict(inner_raw), max_len=int(max_len))
        if isinstance(raw, dict) and "enum" in raw:
            spec = raw["enum"]
            if isinstance(spec, list):
                values = [str(v) for v in spec]
                return cls(tag="enum", values=values, default=values[0] if values else None)
            values = [str(v) for v in spec.get("values", [])]
            if not values:
                raise ValueError(f"enum dtype requires values: {raw}")
            default = spec.get("default")
            oov = spec.get("oov")
            return cls(
                tag="enum",
                values=values,
                default=str(default) if default is not None else values[0],
                oov=str(oov) if oov is not None else None,
            )
        raise ValueError(f"Invalid DType: {raw}")


def parse_int_strict(raw: str) -> int:
    text = raw.strip()
    if not text:
        raise ValueError("empty integer value")
    return int(text)


def parse_float_strict(raw: str) -> float:
    text = raw.strip()
    if not text:
        raise ValueError("empty float value")
    return float(text)


@dataclass
class EmbedConfig:
    vocab_size: int
    embed_dim: int
    pooling: str = "first"  # first | flatten | mean | sum | max
    seq_len: Optional[int] = None  # pooling=flatten 时的序列长度
    truncation: str = "head"  # head | tail


@dataclass
class SourceDef:
    name: str
    dtype: DType
    default_val: str
    source: Optional[str] = None
    embed: Optional[EmbedConfig] = None
    role: str = Role.FEATURE
    column_index: Optional[int] = None


@dataclass
class OperatorDef:
    name: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    embed: Optional[EmbedConfig] = None


@dataclass
class FlowConfig:
    version: str
    sources: list[SourceDef]
    operators: list[OperatorDef]

    @classmethod
    def from_yaml(cls, path: str) -> "FlowConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "FlowConfig":
        sources = []
        for s in raw.get("sources", []):
            embed = EmbedConfig(**s["embed"]) if "embed" in s else None
            sources.append(
                SourceDef(
                    name=s["name"],
                    source=s.get("source"),
                    dtype=DType.from_dict(s["dtype"]),
                    default_val=s["default_val"],
                    embed=embed,
                    role=s.get("role", Role.FEATURE),
                    column_index=s.get("column_index"),
                )
            )
        operators = []
        for o in raw.get("operators", []):
            embed = EmbedConfig(**o["embed"]) if "embed" in o else None
            operators.append(
                OperatorDef(
                    name=o["name"],
                    op_type=o["op_type"],
                    inputs=o.get("inputs", []),
                    outputs=o.get("outputs", []),
                    params=o.get("params", {}),
                    embed=embed,
                )
            )
        return cls(version=raw["version"], sources=sources, operators=operators)

    @property
    def feature_sources(self) -> list[SourceDef]:
        """返回 role==feature 的 source 列表。"""
        return [s for s in self.sources if s.role == Role.FEATURE]

    @property
    def label_sources(self) -> list[SourceDef]:
        """返回 role==label 的 source 列表。"""
        return [s for s in self.sources if s.role == Role.LABEL]

    @property
    def discard_sources(self) -> list[SourceDef]:
        """返回 role==discard 的 source 列表。"""
        return [s for s in self.sources if s.role == Role.DISCARD]


@dataclass
class EvalConfig:
    metrics: list[str] = field(default_factory=lambda: ["auc"])
    monitor_metric: str = "auc"
    log_path: str = ""
    gauc_group_feature: str = "user_id"

    def __post_init__(self) -> None:
        if isinstance(self.metrics, str):
            self.metrics = [m.strip() for m in self.metrics.split(",") if m.strip()]


@dataclass
class OptimConfig:
    name: str = "adamw"
    lr: float = 0.005
    weight_decay: float = 1e-4
    momentum: float = 0.0
    emb_lr: float | None = None
    emb_weight_decay: float | None = None


@dataclass
class LRScheduleConfig:
    warmup_steps: int = 200
    min_lr_ratio: float = 0.01


@dataclass
class ArtifactConfig:
    artifact_root: str = ""
    model_name: str = ""
    run_version: str = ""
    keep_checkpoints: int = 3
    publish_best: bool = True
    publish_latest: bool = True
    copy_configs: bool = False


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    export_path: str = ""
    artifacts: ArtifactConfig | dict[str, Any] = field(default_factory=ArtifactConfig)
    optim: OptimConfig | dict[str, Any] = field(default_factory=OptimConfig)
    lr_schedule: LRScheduleConfig | dict[str, Any] = field(default_factory=LRScheduleConfig)
    eval: EvalConfig | dict[str, Any] = field(default_factory=EvalConfig)
    eval_samples: int = 2000
    eval_interval: int = 50
    log_interval: int = 10
    loss_weighting: str = "static"
    task_weights: dict[str, float] | None = None
    tasks: list[TaskSpec] | list[dict[str, Any]] = field(default_factory=list)
    grad_max_norm: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 5
    tb_dir: str = ""
    tb_grad_interval: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.artifacts, dict):
            self.artifacts = ArtifactConfig(**self.artifacts)
        if isinstance(self.optim, dict):
            self.optim = OptimConfig(**self.optim)
        if isinstance(self.lr_schedule, dict):
            self.lr_schedule = LRScheduleConfig(**self.lr_schedule)
        if isinstance(self.eval, dict):
            self.eval = EvalConfig(**self.eval)
        if self.tasks and not isinstance(self.tasks[0], TaskSpec):
            self.tasks = parse_task_specs(self.tasks)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

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


@dataclass
class ModelConfig:
    """模型配置：type + params，不再兼容旧的散装 kwargs。"""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        mtype = raw["type"]
        params = {k: v for k, v in raw.items() if k != "type"}
        return cls(type=mtype, params=params)

    def build(
        self,
        features: list[FeatureTuple],
        tokenizer: nn.Module | None = None,
        pooling_map: dict[str, str] | None = None,
        total_dim: int | None = None,
    ) -> nn.Module:
        from ..models import build_model

        params = self.params.copy()
        if pooling_map:
            params["_pooling_map"] = pooling_map
        if total_dim is not None:
            params["_total_dim"] = total_dim
        return build_model(self.type, features, tokenizer=tokenizer, **params)
