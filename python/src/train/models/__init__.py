from __future__ import annotations

"""模型注册表：@register_model 装饰器 + 中央 build() 工厂。"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch.nn as nn

from ..layers.embedding import FeatureTuple
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TowerConfig
from ..core.task import label_map as task_label_map
from ..core.task import parse_task_specs, task_names
from .deepfm import DeepFM
from .esmm import ESMM, default_task_config
from .gdcn_esmm import GDCNESMM
from .lr import LogisticRegression
from .mmoe import MMoE
from .unimixer.model import UniMixerModel

# ── registry ──
OutputSpec = dict[str, Any]
BuildFn = Callable[[list[FeatureTuple], Optional[nn.Module]], nn.Module]
OutputSpecFn = Callable[[Optional[nn.Module], Optional[dict[str, Any]]], OutputSpec]

_registry: dict[str, dict[str, Callable[..., Any]]] = {}
"""Each entry: {build_fn, output_spec_fn}"""


def register_model(name: str, output_spec_fn: OutputSpecFn, build_fn: BuildFn) -> None:
    _registry[name] = {"build": build_fn, "output_spec": output_spec_fn}


def build_model(
    model_type: str,
    features: list[FeatureTuple],
    tokenizer: Optional[nn.Module] = None,
    **params: Any,
) -> nn.Module:
    """Build any registered model by type name. No if-elif chain."""
    if model_type not in _registry:
        raise ValueError(f"Unknown model type: {model_type}. Registered: {list(_registry)}")
    return _registry[model_type]["build"](features, tokenizer, **params)


def get_output_spec(
    model_type: str,
    model_instance: Optional[nn.Module] = None,
    params: Optional[dict[str, Any]] = None,
) -> OutputSpec:
    """Get output spec dict {task_names, label_col_map} for a model type."""
    if model_type not in _registry:
        return {"task_names": [], "label_col_map": {}}
    fn = _registry[model_type]["output_spec"]
    try:
        return fn(model_instance, params=params)
    except TypeError:
        return fn(model_instance)


@dataclass
class TaskConfigEntry:
    name: str
    tower_dims: list[int] = field(default_factory=list)


def _parse_task_config(raw: Optional[dict[str, Any]]) -> Optional[MultiTaskConfig]:
    if not raw:
        return None
    towers = [
        TowerConfig(
            t["name"],
            t.get("hidden_dims", []),
            t.get("output_dim", 1),
            Activation.from_str(t.get("activation", "relu")),
        )
        for t in raw.get("towers", [])
    ]
    relations = [TaskRelation(r["target"], r["sources"], r["op"]) for r in raw.get("relations", [])]
    return MultiTaskConfig(towers=towers, relations=relations)


def _default_esmm_task_config(params: dict[str, Any]) -> MultiTaskConfig:
    return default_task_config(
        params.get("click_hidden_dims", [8]),
        params.get("cvr_hidden_dims", [8]),
        params.get("detail_hidden_dims", [8]),
        params.get("stock_hidden_dims", [8]),
        params.get("stay_hidden_dims", [8]),
    )


def _parse_mmoe_task_configs(raw: dict[str, Any]) -> list[TaskConfigEntry]:
    return [
        TaskConfigEntry(t["name"], t.get("tower_dims", [])) for t in raw.get("task_configs", [])
    ]


# ── register built-in models ──


def _spec_pred(model: Optional[nn.Module] = None, params: Optional[dict[str, Any]] = None) -> OutputSpec:
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    return {"task_names": ["pred"], "label_col_map": {"pred": "is_click"}}


def _build_lr(
    features: list[FeatureTuple], tokenizer: Optional[nn.Module] = None, **params: Any
) -> LogisticRegression:
    return LogisticRegression(
        features, pooling_map=params.get("_pooling_map"), total_dim=params.get("_total_dim")
    )


def _build_deepfm(
    features: list[FeatureTuple], tokenizer: Optional[nn.Module] = None, **params: Any
) -> DeepFM:
    return DeepFM(
        features,
        params.get("fm_k", 16),
        params.get("deep_hidden_dims", []),
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
    )


def _build_mmoe(
    features: list[FeatureTuple], tokenizer: Optional[nn.Module] = None, **params: Any
) -> MMoE:
    tcs = [(t.name, t.tower_dims) for t in _parse_mmoe_task_configs(params)]
    return MMoE(
        features,
        params.get("shared_bottom_dims", []),
        params.get("num_experts", 4),
        params.get("expert_hidden_dims", []),
        params.get("expert_output_dim", 32),
        tcs,
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
    )


def _spec_mmoe(model: Optional[nn.Module] = None, params: Optional[dict[str, Any]] = None) -> OutputSpec:
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    names = model.task_names if model else []
    return {"task_names": names, "label_col_map": {n: n for n in names}}


def _build_esmm(
    features: list[FeatureTuple], tokenizer: Optional[nn.Module] = None, **params: Any
) -> ESMM:
    task_config = _parse_task_config(params.get("task_config")) or _default_esmm_task_config(params)
    return ESMM(
        features,
        params.get("shared_bottom_dims", []),
        params.get("click_hidden_dims", [8]),
        params.get("cvr_hidden_dims", [8]),
        params.get("detail_hidden_dims", [8]),
        params.get("stock_hidden_dims", [8]),
        params.get("stay_hidden_dims", [8]),
        task_config=task_config,
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
    )


def _build_gdcn_esmm(
    features: list[FeatureTuple], tokenizer: Optional[nn.Module] = None, **params: Any
) -> GDCNESMM:
    task_config = _parse_task_config(params.get("task_config")) or _default_esmm_task_config(params)
    return GDCNESMM(
        features,
        cross_layers=params.get("cross_layers", 3),
        deep_hidden_dims=params.get("deep_hidden_dims", []),
        shared_bottom_dims=params.get("shared_bottom_dims", []),
        click_hidden_dims=params.get("click_hidden_dims", [8]),
        cvr_hidden_dims=params.get("cvr_hidden_dims", [8]),
        detail_hidden_dims=params.get("detail_hidden_dims", [8]),
        stock_hidden_dims=params.get("stock_hidden_dims", [8]),
        stay_hidden_dims=params.get("stay_hidden_dims", [8]),
        task_config=task_config,
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
    )


def _spec_esmm(model: Optional[nn.Module] = None, params: Optional[dict[str, Any]] = None) -> OutputSpec:
    params = params or {}
    specs = parse_task_specs(params.get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    if model is not None:
        names = list(getattr(model, "task_names", []))
    else:
        task_config = _parse_task_config(params.get("task_config")) or _default_esmm_task_config(
            params
        )
        names = [tower.name for tower in task_config.towers]
    label_col_map = params.get(
        "label_col_map",
        {
            "click": "is_click",
            "cvr": "is_cvr",
            "detail": "is_click_detail",
            "stock": "is_click_stock",
            "stay": "stay_time",
        },
    )
    return {"task_names": names, "label_col_map": label_col_map}


def _spec_unimixer(
    model: Optional[nn.Module] = None, params: Optional[dict[str, Any]] = None
) -> OutputSpec:
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    if model is not None:
        tt = getattr(model, "task_towers", None) or model.unimixer.task_towers
        names = list(tt._tower_names) if hasattr(tt, "_tower_names") else []
    else:
        names = []
    label_col_map = (params or {}).get("label_col_map", {n: n for n in names} if names else {})
    return {"task_names": names, "label_col_map": label_col_map}


def _build_unimixer(
    features: list[FeatureTuple], tokenizer: Optional[nn.Module] = None, **params: Any
) -> UniMixerModel:
    if tokenizer is None:
        raise ValueError("UniMixer requires external FeatureTokenizer")
    tc = _parse_task_config(params.get("task_config"))
    if tc is None:
        raise ValueError("UniMixer requires task_config")
    return UniMixerModel(
        tokenizer=tokenizer,
        token_dim=params.get("token_dim", 64),
        num_tokens=params.get("num_tokens", 8),
        num_blocks=params.get("num_blocks", 2),
        block_size_opt=params.get("block_size"),
        use_lite=params.get("use_lite", False),
        hidden_factor=params.get("hidden_factor", 1.0),
        num_basis=params.get("num_basis", 4),
        rank=params.get("rank", 16),
        task_config=tc,
        use_siamese=params.get("use_siamese", False),
    )


register_model("lr", _spec_pred, _build_lr)
register_model("deepfm", _spec_pred, _build_deepfm)
register_model("mmoe", _spec_mmoe, _build_mmoe)
register_model("esmm", _spec_esmm, _build_esmm)
register_model("gdcn_esmm", _spec_esmm, _build_gdcn_esmm)
register_model("unimixer", _spec_unimixer, _build_unimixer)
