from __future__ import annotations

"""训练与推理配置的单一入口。"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch.nn as nn
import yaml

from ..layers.embedding import FeatureTuple
from .task import TaskSpec, parse_task_specs


class Role:
    """列角色常量 — mirrors src/feats/config.rs Role enum."""

    FEATURE = "feature"
    LABEL = "label"
    DISCARD = "discard"


class SourceKind(str, Enum):
    USER = "User"
    ITEM = "Item"
    CONTEXT = "Context"


@dataclass
class DType:
    tag: DTypeTag
    inner: DType | None = None
    max_len: int | None = None
    values: list[str] | None = None
    default: str | None = None
    oov: str | None = None

    @property
    def length(self) -> int:
        return self.max_len or 0

    @classmethod
    def from_dict(cls, raw: str | dict) -> DType:
        if isinstance(raw, str):
            return cls(tag=DTypeTag(raw))
        if isinstance(raw, dict) and "list" in raw:
            spec = raw["list"]
            inner_raw = spec.get("item_dtype", spec.get("dtype"))
            max_len = spec.get("max_len", spec.get("length"))
            if inner_raw is None:
                raise ValueError(f"Invalid list DType: {raw}")
            if max_len is None:
                raise ValueError(f"list dtype requires max_len: {raw}")
            return cls(tag=DTypeTag.LIST, inner=cls.from_dict(inner_raw), max_len=int(max_len))
        if isinstance(raw, dict) and "enum" in raw:
            spec = raw["enum"]
            if isinstance(spec, list):
                values = [str(v) for v in spec]
                return cls(tag=DTypeTag.ENUM, values=values, default=values[0] if values else None)
            values = [str(v) for v in spec.get("values", [])]
            if not values:
                raise ValueError(f"enum dtype requires values: {raw}")
            default = spec.get("default")
            oov = spec.get("oov")
            return cls(
                tag=DTypeTag.ENUM,
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


class PoolingMode(str, Enum):
    FIRST = "first"
    FLATTEN = "flatten"
    MEAN = "mean"
    SUM = "sum"
    MAX = "max"


class TruncationSide(str, Enum):
    HEAD = "head"
    TAIL = "tail"


class CrossType(str, Enum):
    INNER_PRODUCT = "inner_product"
    CARTESIAN = "cartesian"


class ParseMode(str, Enum):
    JSON = "json"
    STRUCTURED = "structured"
    STRUCTURED_FLAT_SPLIT = "structured_flat_split"
    STRUCTURED_LIST_SPLIT = "structured_list_split"
    SPLIT = "split"
    LIST_SPLIT = "list_split"
    FLAT_SPLIT = "flat_split"


@dataclass
class EmbedConfig:
    vocab_size: int
    embed_dim: int
    pooling: PoolingMode = PoolingMode.FIRST
    seq_len: int | None = None
    truncation: TruncationSide = TruncationSide.HEAD

    def __post_init__(self) -> None:
        if isinstance(self.pooling, str):
            object.__setattr__(self, "pooling", PoolingMode(self.pooling))
        if isinstance(self.truncation, str):
            object.__setattr__(self, "truncation", TruncationSide(self.truncation))


@dataclass
class SourceDef:
    name: str
    dtype: DType
    default_val: str
    source: SourceKind | None = None
    data_source: str | None = None
    embed: EmbedConfig | None = None
    role: str = Role.FEATURE
    column_index: int | None = None


class DTypeTag(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    ENUM = "enum"
    LIST = "list"


class OpType(str, Enum):
    BUCKETING = "Bucketing"
    CONCAT_HASH = "ConcatHash"
    CROSS_FEATURE = "CrossFeature"
    DICT_MAPPER = "DictMapper"
    EXPRESSION_OP = "ExpressionOp"
    FEATURE_HASH = "FeatureHash"
    FLAT_SPLIT = "FlatSplit"
    JSON_EXTRACT_LIST = "JsonExtractList"
    LIST_OVERLAP = "ListOverlap"
    LIST_STRING_PARSER = "ListStringParser"
    PARSED_FEATURE_HASH = "ParsedFeatureHash"
    PLUGIN_OP = "PluginOp"
    SEQUENCE_OP = "SequenceOp"
    SPLIT = "Split"
    STRING_CONCAT = "StringConcat"
    STRING_PARSER = "StringParser"


_OP_PARAM_SPECS: dict[OpType, tuple[set[str], set[str], dict[str, type | tuple[type, ...]]]] = {
    OpType.BUCKETING: ({"boundaries"}, {"boundaries"}, {"boundaries": list}),
    OpType.CONCAT_HASH: (
        {"vocab_size", "num_hashes", "separator", "namespace", "salt", "version"},
        {"vocab_size"},
        {
            "vocab_size": int,
            "num_hashes": int,
            "separator": str,
            "namespace": str,
            "salt": str,
            "version": str,
        },
    ),
    OpType.CROSS_FEATURE: ({"cross_type", "max_len"}, set(), {"cross_type": str, "max_len": int}),
    OpType.DICT_MAPPER: (
        {"mapping", "default_idx"},
        {"mapping"},
        {"mapping": dict, "default_idx": int},
    ),
    OpType.EXPRESSION_OP: ({"script"}, {"script"}, {"script": str}),
    OpType.FEATURE_HASH: (
        {"vocab_size", "num_hashes", "separator", "namespace", "salt", "version"},
        {"vocab_size"},
        {
            "vocab_size": int,
            "num_hashes": int,
            "separator": str,
            "namespace": str,
            "salt": str,
            "version": str,
        },
    ),
    OpType.FLAT_SPLIT: (
        {"sep", "max_len", "pad_val"},
        set(),
        {"sep": str, "max_len": int, "pad_val": str},
    ),
    OpType.JSON_EXTRACT_LIST: (
        {"key", "pad_len", "pad_val"},
        set(),
        {"key": str, "pad_len": int, "pad_val": str},
    ),
    OpType.LIST_OVERLAP: (set(), set(), {}),
    OpType.LIST_STRING_PARSER: ({"sep", "key_index"}, set(), {"sep": str, "key_index": int}),
    OpType.PARSED_FEATURE_HASH: (
        {
            "vocab_size",
            "parse_mode",
            "num_hashes",
            "separator",
            "namespace",
            "salt",
            "version",
            "key",
            "sep1",
            "sep2",
            "key_index",
            "sep",
            "max_len",
            "pad_len",
            "pad_val",
        },
        {"vocab_size"},
        {
            "vocab_size": int,
            "parse_mode": str,
            "num_hashes": int,
            "separator": str,
            "namespace": str,
            "salt": str,
            "version": str,
            "key": str,
            "sep1": str,
            "sep2": str,
            "key_index": int,
            "sep": str,
            "max_len": int,
            "pad_len": int,
            "pad_val": str,
        },
    ),
    OpType.PLUGIN_OP: (
        {"path", "lib", "symbol", "args"},
        set(),
        {"path": str, "lib": str, "symbol": str, "args": dict},
    ),
    OpType.SEQUENCE_OP: ({"max_len", "pad_val"}, {"max_len"}, {"max_len": int, "pad_val": int}),
    OpType.SPLIT: (
        {"sep", "max_len", "pad_val"},
        set(),
        {"sep": str, "max_len": int, "pad_val": str},
    ),
    OpType.STRING_CONCAT: ({"separator"}, set(), {"separator": str}),
    OpType.STRING_PARSER: (
        {"sep1", "sep2", "key_index", "pad_len", "pad_val"},
        set(),
        {"sep1": str, "sep2": str, "key_index": int, "pad_len": int, "pad_val": str},
    ),
}


_COMMON_MODEL_KEYS = {"tasks", "label_col_map", "metrics"}
_MODEL_PARAM_SPECS: dict[str, tuple[set[str], set[str], dict[str, type | tuple[type, ...]]]] = {
    "lr": (_COMMON_MODEL_KEYS, set(), {"tasks": list, "label_col_map": dict, "metrics": dict}),
    "deepfm": (
        _COMMON_MODEL_KEYS | {"fm_k", "deep_hidden_dims"},
        set(),
        {
            "fm_k": int,
            "deep_hidden_dims": list,
            "tasks": list,
            "label_col_map": dict,
            "metrics": dict,
        },
    ),
    "mmoe": (
        _COMMON_MODEL_KEYS
        | {
            "shared_bottom_dims",
            "num_experts",
            "expert_hidden_dims",
            "expert_output_dim",
            "task_configs",
        },
        set(),
        {
            "shared_bottom_dims": list,
            "num_experts": int,
            "expert_hidden_dims": list,
            "expert_output_dim": int,
            "task_configs": list,
            "tasks": list,
            "label_col_map": dict,
            "metrics": dict,
        },
    ),
    "esmm": (
        _COMMON_MODEL_KEYS
        | {
            "shared_bottom_dims",
            "click_hidden_dims",
            "cvr_hidden_dims",
            "detail_hidden_dims",
            "stock_hidden_dims",
            "stay_hidden_dims",
            "task_config",
        },
        set(),
        {
            "shared_bottom_dims": list,
            "click_hidden_dims": list,
            "cvr_hidden_dims": list,
            "detail_hidden_dims": list,
            "stock_hidden_dims": list,
            "stay_hidden_dims": list,
            "task_config": dict,
            "tasks": list,
            "label_col_map": dict,
            "metrics": dict,
        },
    ),
    "gdcn_esmm": (
        _COMMON_MODEL_KEYS
        | {
            "cross_layers",
            "deep_hidden_dims",
            "shared_bottom_dims",
            "click_hidden_dims",
            "cvr_hidden_dims",
            "detail_hidden_dims",
            "stock_hidden_dims",
            "stay_hidden_dims",
            "task_config",
        },
        set(),
        {
            "cross_layers": int,
            "deep_hidden_dims": list,
            "shared_bottom_dims": list,
            "click_hidden_dims": list,
            "cvr_hidden_dims": list,
            "detail_hidden_dims": list,
            "stock_hidden_dims": list,
            "stay_hidden_dims": list,
            "task_config": dict,
            "tasks": list,
            "label_col_map": dict,
            "metrics": dict,
        },
    ),
    "unimixer": (
        _COMMON_MODEL_KEYS
        | {
            "token_dim",
            "num_tokens",
            "num_blocks",
            "block_size",
            "use_lite",
            "hidden_factor",
            "num_basis",
            "rank",
            "task_config",
            "use_siamese",
        },
        {"task_config"},
        {
            "token_dim": int,
            "num_tokens": int,
            "num_blocks": int,
            "block_size": int,
            "use_lite": bool,
            "hidden_factor": (int, float),
            "num_basis": int,
            "rank": int,
            "task_config": dict,
            "use_siamese": bool,
            "tasks": list,
            "label_col_map": dict,
            "metrics": dict,
        },
    ),
    "token_mixer_large": (
        _COMMON_MODEL_KEYS
        | {
            "token_dim",
            "num_tokens",
            "num_blocks",
            "num_heads",
            "hidden_factor",
            "task_config",
            "down_init_scale",
        },
        {"task_config"},
        {
            "token_dim": int,
            "num_tokens": int,
            "num_blocks": int,
            "num_heads": int,
            "hidden_factor": (int, float),
            "task_config": dict,
            "down_init_scale": (int, float),
            "tasks": list,
            "label_col_map": dict,
            "metrics": dict,
        },
    ),
}


def _validate_mapping_keys(
    context: str, raw: dict[str, Any], allowed: set[str], required: set[str]
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {sorted(unknown)}")
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{context} missing required field(s): {sorted(missing)}")


def _validate_typed_params(
    context: str,
    params: dict[str, Any],
    allowed: set[str],
    required: set[str],
    types: dict[str, type | tuple[type, ...]],
) -> None:
    _validate_mapping_keys(context, params, allowed, required)
    for key, expected in types.items():
        if key not in params or params[key] is None:
            continue
        value = params[key]
        if isinstance(value, bool) and (
            expected is int
            or (isinstance(expected, tuple) and int in expected and bool not in expected)
        ):
            raise ValueError(f"{context}.{key} must be int, got bool")
        if not isinstance(value, expected):
            names = (
                ", ".join(t.__name__ for t in expected)
                if isinstance(expected, tuple)
                else expected.__name__
            )
            raise TypeError(f"{context}.{key} must be {names}, got {type(value).__name__}")


@dataclass
class OperatorDef:
    name: str
    op_type: OpType
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    embed: EmbedConfig | None = None


@dataclass
class DataSourceDef:
    name: str
    kind: str
    description: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowConfig:
    version: str
    data_sources: list[DataSourceDef]
    sources: list[SourceDef]
    operators: list[OperatorDef]

    @classmethod
    def from_yaml(cls, path: str) -> FlowConfig:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> FlowConfig:
        _validate_mapping_keys(
            "FlowConfig",
            raw,
            {"version", "data_sources", "sources", "operators"},
            {"version", "sources", "operators"},
        )
        data_sources = [
            _validate_mapping_keys(
                f"data_sources[{idx}]",
                ds,
                {"name", "kind", "description", "params"},
                {"name", "kind"},
            )
            or DataSourceDef(
                name=str(ds["name"]),
                kind=str(ds["kind"]),
                description=ds.get("description"),
                params=ds.get("params", {}),
            )
            for idx, ds in enumerate(raw.get("data_sources", []))
        ]
        sources = []
        for idx, s in enumerate(raw.get("sources", [])):
            _validate_mapping_keys(
                f"sources[{idx}]",
                s,
                {
                    "name",
                    "source",
                    "data_source",
                    "dtype",
                    "default_val",
                    "embed",
                    "role",
                    "column_index",
                },
                {"name", "dtype", "default_val"},
            )
            embed = EmbedConfig(**s["embed"]) if "embed" in s else None
            sources.append(
                SourceDef(
                    name=s["name"],
                    source=SourceKind(s["source"]) if s.get("source") else None,
                    data_source=s.get("data_source"),
                    dtype=DType.from_dict(s["dtype"]),
                    default_val=s["default_val"],
                    embed=embed,
                    role=s.get("role", Role.FEATURE),
                    column_index=s.get("column_index"),
                )
            )
        data_source_names = {source.name for source in data_sources}
        for source in sources:
            if source.data_source and source.data_source not in data_source_names:
                raise ValueError(
                    f"source '{source.name}' references unknown data_source '{source.data_source}'"
                )
        operators = []
        for idx, o in enumerate(raw.get("operators", [])):
            _validate_mapping_keys(
                f"operators[{idx}]",
                o,
                {"name", "op_type", "inputs", "outputs", "params", "embed"},
                {"name", "op_type", "inputs", "outputs"},
            )
            op_type = OpType(o["op_type"])
            allowed, required, types = _OP_PARAM_SPECS[op_type]
            _validate_typed_params(
                f"operator '{o['name']}' params", o.get("params", {}), allowed, required, types
            )
            embed = EmbedConfig(**o["embed"]) if "embed" in o else None
            operators.append(
                OperatorDef(
                    name=o["name"],
                    op_type=op_type,
                    inputs=o.get("inputs", []),
                    outputs=o.get("outputs", []),
                    params=o.get("params", {}),
                    embed=embed,
                )
            )
        return cls(
            version=raw["version"],
            data_sources=data_sources,
            sources=sources,
            operators=operators,
        )

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
    monitor_task: str = ""
    monitor_mode: str = "auto"
    log_path: str = ""
    gauc_group_feature: str = "user_id"

    def __post_init__(self) -> None:
        if isinstance(self.metrics, str):
            self.metrics = [m.strip() for m in self.metrics.split(",") if m.strip()]
        if self.monitor_mode not in {"auto", "max", "min"}:
            raise ValueError(f"eval.monitor_mode must be auto, max, or min: {self.monitor_mode}")


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
    copy_configs: bool = True


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    export_path: str = ""
    prefetch_batches: int = 2
    checkpoint_interval_steps: int = 0
    checkpoint_interval_seconds: float = 0.0
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
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        return cls(**raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open(encoding="utf-8") as f:
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
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with Path(path).open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        _validate_mapping_keys("ModelConfig", raw, {"type"} | set(raw.keys()), {"type"})
        mtype = raw["type"]
        params = {k: v for k, v in raw.items() if k != "type"}
        if mtype not in _MODEL_PARAM_SPECS:
            raise ValueError(
                f"Unknown model type: {mtype}. Registered: {sorted(_MODEL_PARAM_SPECS)}"
            )
        allowed, required, types = _MODEL_PARAM_SPECS[mtype]
        _validate_typed_params(f"model '{mtype}' params", params, allowed, required, types)
        return cls(type=mtype, params=params)

    def build(
        self,
        features: list[FeatureTuple],
        tokenizer: nn.Module | None = None,
        pooling_map: dict[str, PoolingMode] | None = None,
        total_dim: int | None = None,
    ) -> nn.Module:
        from ..models import build_model

        params = self.params.copy()
        if pooling_map:
            params["_pooling_map"] = pooling_map
        if total_dim is not None:
            params["_total_dim"] = total_dim
        return build_model(self.type, features, tokenizer=tokenizer, **params)
