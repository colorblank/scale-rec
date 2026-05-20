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
from .deepfm import DeepFM
from .esmm import ESMM
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


def get_output_spec(model_type: str, model_instance=None):
    """Get output spec dict {task_names, label_col_map} for a model type."""
    if model_type not in _registry:
        return {"task_names": [], "label_col_map": {}}
    return _registry[model_type]["output_spec"](model_instance)


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

    def build(self, features, tokenizer=None):
        return build_model(self.type, features, tokenizer=tokenizer, **self.params)


# ── register built-in models ──


def _spec_pred(model=None):
    return {"task_names": ["pred"], "label_col_map": {"pred": "is_click"}}


def _build_lr(features, tokenizer=None, **params):
    return LogisticRegression(features)


def _build_deepfm(features, tokenizer=None, **params):
    return DeepFM(features, params.get("fm_k", 16), params.get("deep_hidden_dims", []))


def _build_mmoe(features, tokenizer=None, **params):
    tcs = [(t.name, t.tower_dims) for t in _parse_mmoe_task_configs(params)]
    return MMoE(
        features,
        params.get("shared_bottom_dims", []),
        params.get("num_experts", 4),
        params.get("expert_hidden_dims", []),
        params.get("expert_output_dim", 32),
        tcs,
    )


def _spec_mmoe(model=None):
    names = model.task_names if model else []
    return {"task_names": names, "label_col_map": {n: n for n in names}}


def _build_esmm(features, tokenizer=None, **params):
    return ESMM(
        features,
        params.get("shared_bottom_dims", []),
        params.get("ctr_hidden_dims", []),
        params.get("cvr_hidden_dims", []),
    )


def _spec_esmm(model=None):
    return {"task_names": ["ctr", "cvr"], "label_col_map": {"ctr": "is_click", "cvr": "is_cvr"}}


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


def _spec_unimixer(model=None):
    if model is not None:
        tt = getattr(model, "task_towers", None) or model.unimixer.task_towers
        names = list(tt._tower_names) if hasattr(tt, "_tower_names") else []
    else:
        names = []
    return {"task_names": names, "label_col_map": {n: n for n in names}}


register_model("lr", _spec_pred, _build_lr)
register_model("deepfm", _spec_pred, _build_deepfm)
register_model("mmoe", _spec_mmoe, _build_mmoe)
register_model("esmm", _spec_esmm, _build_esmm)
register_model("unimixer", _spec_unimixer, _build_unimixer)
