from __future__ import annotations

"""模型注册表：@register_model 装饰器 + 中央 build() 工厂。

新增模型只需:
1. 创建 models/newmodel.py，用 @register_model("newmodel") 装饰
2. 实现 output_spec() 返回 {task_names, label_col_map}
3. 实现 from_params(params) 静态方法解析 YAML params
"""
from dataclasses import dataclass, field

import yaml

from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TowerConfig
from ..task import label_map as task_label_map
from ..task import parse_task_specs, task_names
from .deepfm import DeepFM
from .esmm import ESMM, default_task_config
from .lr import LogisticRegression
from .mmoe import MMoE
from .unimixer.model import UniMixerModel

# ── registry ──
_registry: dict[str, dict] = {}
"""Each entry: {build_fn, output_spec_fn}"""


def register_model(name: str, output_spec_fn, build_fn):
    _registry[name] = {"build": build_fn, "output_spec": output_spec_fn}


def build_model(model_type: str, features, tokenizer=None, **params):
    """Build any registered model by type name. No if-elif chain."""
    if model_type not in _registry:
        raise ValueError(f"Unknown model type: {model_type}. Registered: {list(_registry)}")
    return _registry[model_type]["build"](features, tokenizer, **params)


def get_output_spec(model_type: str, model_instance=None, params: dict | None = None):
    """Get output spec dict {task_names, label_col_map} for a model type."""
    if model_type not in _registry:
        return {"task_names": [], "label_col_map": {}}
    fn = _registry[model_type]["output_spec"]
    try:
        return fn(model_instance, params=params)
    except TypeError:
        return fn(model_instance)


# ── backward-compat config types ──


@dataclass
class TaskConfigEntry:
    name: str
    tower_dims: list[int] = field(default_factory=list)


def _parse_task_config(raw):
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


def _default_esmm_task_config(params):
    return default_task_config(
        params.get("click_hidden_dims", [8]),
        params.get("cvr_hidden_dims", [8]),
        params.get("detail_hidden_dims", [8]),
        params.get("stock_hidden_dims", [8]),
        params.get("stay_hidden_dims", [8]),
    )


def _parse_mmoe_task_configs(raw):
    return [
        TaskConfigEntry(t["name"], t.get("tower_dims", [])) for t in raw.get("task_configs", [])
    ]


@dataclass
class ModelConfig:
    """YAML model config. Kept minimal: type + raw params dict."""

    type: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw):
        mtype = raw["type"]
        params = {k: v for k, v in raw.items() if k != "type"}
        return cls(type=mtype, params=params)

    def build(self, features, tokenizer=None, pooling_map=None, total_dim=None):
        if pooling_map:
            self.params["_pooling_map"] = pooling_map
        if total_dim is not None:
            self.params["_total_dim"] = total_dim
        return build_model(self.type, features, tokenizer=tokenizer, **self.params)


# ── register built-in models ──


def _spec_pred(model=None, params=None):
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    return {"task_names": ["pred"], "label_col_map": {"pred": "is_click"}}


def _build_lr(features, tokenizer=None, **params):
    return LogisticRegression(
        features, pooling_map=params.get("_pooling_map"), total_dim=params.get("_total_dim")
    )


def _build_deepfm(features, tokenizer=None, **params):
    return DeepFM(
        features,
        params.get("fm_k", 16),
        params.get("deep_hidden_dims", []),
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
    )


def _build_mmoe(features, tokenizer=None, **params):
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


def _spec_mmoe(model=None, params=None):
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    names = model.task_names if model else []
    return {"task_names": names, "label_col_map": {n: n for n in names}}


def _build_esmm(features, tokenizer=None, **params):
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


def _spec_esmm(model=None, params=None):
    params = params or {}
    specs = parse_task_specs(params.get("tasks"))
    if specs:
        return {
            "tasks": specs,
            "task_names": task_names(specs),
            "label_col_map": task_label_map(specs),
        }
    if model is not None:
        task_names = list(getattr(model, "task_names", []))
    else:
        task_config = _parse_task_config(params.get("task_config")) or _default_esmm_task_config(
            params
        )
        task_names = [tower.name for tower in task_config.towers]
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
    return {"task_names": task_names, "label_col_map": label_col_map}


def _spec_unimixer(model=None, params=None):
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


def _build_unimixer(features, tokenizer=None, **params):
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
register_model("unimixer", _spec_unimixer, _build_unimixer)
