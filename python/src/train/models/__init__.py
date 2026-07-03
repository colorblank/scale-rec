from __future__ import annotations

"""模型注册表：@register_model 装饰器 + 中央 build() 工厂。"""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from ..core.task import TaskContract, parse_task_specs
from ..layers.embedding import FeatureTuple
from ..layers.towers import Activation, MultiTaskConfig, TaskRelation, TowerConfig
from .deepfm import DeepFM
from .esmm import ESMM, default_task_config
from .fat import FATModel
from .gdcn_esmm import GDCNESMM
from .lr import LogisticRegression
from .mixformer import MixFormerModel
from .mmoe import MMoE
from .onerank import OneRankModel
from .pepnet import PEPNet
from .rankmixer import RankMixerModel
from .token_mixer_large import TokenMixerLargeModel
from .unimixer.model import UniMixerModel

# ── registry ──
OutputSpec = dict[str, Any]
BuildFn = Callable[[list[FeatureTuple], nn.Module | None], nn.Module]
OutputSpecFn = Callable[[nn.Module | None, dict[str, Any] | None], OutputSpec]

_registry: dict[str, dict[str, Callable[..., Any]]] = {}
"""Each entry: {build_fn, output_spec_fn}"""


def register_model(name: str, output_spec_fn: OutputSpecFn, build_fn: BuildFn) -> None:
    _registry[name] = {"build": build_fn, "output_spec": output_spec_fn}


def build_model(
    model_type: str,
    features: list[FeatureTuple],
    tokenizer: nn.Module | None = None,
    **params: Any,
) -> nn.Module:
    """Build any registered model by type name. No if-elif chain."""
    if model_type not in _registry:
        raise ValueError(f"Unknown model type: {model_type}. Registered: {list(_registry)}")
    return _registry[model_type]["build"](features, tokenizer, **params)


def get_output_spec(
    model_type: str,
    model_instance: nn.Module | None = None,
    params: dict[str, Any] | None = None,
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


def _parse_task_config(raw: dict[str, Any] | None) -> MultiTaskConfig | None:
    if not raw:
        return None
    towers = [
        TowerConfig(
            t["name"],
            t.get("hidden_dims", []),
            t.get("output_dim", 1),
            Activation.from_str(t.get("activation", "relu")),
            t.get("output_kind", t.get("output", "binary_logit")),
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


def _output_kinds(
    task_names: list[str],
    relation_names: list[str] | None = None,
    base_kinds: dict[str, str] | None = None,
) -> dict[str, str]:
    kinds = dict.fromkeys(task_names, "binary_logit")
    kinds.update(base_kinds or {})
    for name in relation_names or []:
        kinds[name] = "probability"
    return kinds


def _parse_output_contract(params: dict[str, Any]):
    raw = params.get("output_contract")
    if raw is None:
        return None
    from ..core.output_contract import parse_output_contract

    return parse_output_contract(raw)


def _native_contract_spec(params: dict[str, Any]) -> OutputSpec | None:
    contract = _parse_output_contract(params)
    if contract is None:
        return None
    task_names = list(dict.fromkeys(metric.source for metric in contract.metrics))
    label_col_map: dict[str, str] = {}
    for item in (*contract.objectives, *contract.metrics):
        previous = label_col_map.get(item.source)
        if previous is not None and previous != item.label:
            raise ValueError(
                f"output node '{item.source}' references conflicting labels "
                f"'{previous}' and '{item.label}'"
            )
        label_col_map[item.source] = item.label
    task_metrics: dict[str, list[str]] = {}
    for metric in contract.metrics:
        task_metrics.setdefault(metric.source, []).append(metric.type)
    return {
        "task_names": task_names,
        "label_col_map": label_col_map,
        "output_kinds": contract.node_kinds,
        "task_metrics": task_metrics,
        "output_contract": contract,
    }


def _contract_spec(specs: list[Any], relation_names: list[str] | None = None) -> OutputSpec:
    contract = TaskContract.from_specs(specs)
    return {
        "tasks": list(contract.specs),
        "task_names": contract.task_names,
        "label_col_map": contract.label_col_map,
        "output_kinds": _output_kinds(contract.task_names, relation_names, contract.output_kinds),
    }


# ── register built-in models ──


def _spec_pred(model: nn.Module | None = None, params: dict[str, Any] | None = None) -> OutputSpec:
    native = _native_contract_spec(params or {})
    if native is not None:
        return native
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return _contract_spec(specs)
    return {
        "task_names": ["pred"],
        "label_col_map": {"pred": "is_click"},
        "output_kinds": {"pred": "binary_logit"},
    }


def _build_lr(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> LogisticRegression:
    return LogisticRegression(
        features,
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
        output_contract=_parse_output_contract(params),
    )


def _build_deepfm(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> DeepFM:
    return DeepFM(
        features,
        params.get("fm_k", 16),
        params.get("deep_hidden_dims", []),
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
        output_contract=_parse_output_contract(params),
    )


def _build_mmoe(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> MMoE:
    output_contract = _parse_output_contract(params)
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
        output_contract=output_contract,
    )


def _spec_mmoe(model: nn.Module | None = None, params: dict[str, Any] | None = None) -> OutputSpec:
    native = _native_contract_spec(params or {})
    if native is not None:
        return native
    specs = parse_task_specs((params or {}).get("tasks"))
    if specs:
        return _contract_spec(specs)
    names = model.task_names if model else []
    return {
        "task_names": names,
        "label_col_map": {n: n for n in names},
        "output_kinds": _output_kinds(names),
    }


def _build_esmm(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> ESMM:
    output_contract = _parse_output_contract(params)
    task_config = _parse_task_config(params.get("task_config"))
    if task_config is None and output_contract is None:
        raise ValueError("ESMM requires task_config")
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
        output_contract=output_contract,
    )


def _build_gdcn_esmm(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> GDCNESMM:
    output_contract = _parse_output_contract(params)
    task_config = _parse_task_config(params.get("task_config"))
    if task_config is None and output_contract is None:
        raise ValueError("GDCNESMM requires task_config or output_contract")
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
        output_contract=output_contract,
    )


def _spec_esmm(model: nn.Module | None = None, params: dict[str, Any] | None = None) -> OutputSpec:
    params = params or {}
    native = _native_contract_spec(params)
    if native is not None:
        return native
    task_config = _parse_task_config(params.get("task_config")) or _default_esmm_task_config(params)
    relation_names = [relation.target for relation in task_config.relations]
    specs = parse_task_specs(params.get("tasks"))
    if specs:
        return _contract_spec(specs, relation_names)
    if model is not None:
        names = list(getattr(model, "task_names", []))
        base_kinds = {
            tower.name: tower.output_kind
            for tower in getattr(model, "task_config", task_config).towers
        }
    else:
        names = [tower.name for tower in task_config.towers]
        base_kinds = {tower.name: tower.output_kind for tower in task_config.towers}
    label_col_map = params.get(
        "label_col_map",
        {
            "click": "is_click",
            "cvr": "is_cvr",
            "detail": "is_click_detail",
            "stock": "is_click_stock",
            "stay": "stay_time_label",
        },
    )
    return {
        "task_names": names,
        "label_col_map": label_col_map,
        "output_kinds": _output_kinds(names, relation_names, base_kinds),
    }


def _spec_unimixer(
    model: nn.Module | None = None, params: dict[str, Any] | None = None
) -> OutputSpec:
    native = _native_contract_spec(params or {})
    if native is not None:
        return native
    specs = parse_task_specs((params or {}).get("tasks"))
    relation_names: list[str] = []
    task_config = _parse_task_config((params or {}).get("task_config"))
    if task_config is not None:
        relation_names = [relation.target for relation in task_config.relations]
    if specs:
        return _contract_spec(specs, relation_names)
    if model is not None:
        tt = getattr(model, "task_towers", None) or model.unimixer.task_towers
        names = list(tt._tower_names) if hasattr(tt, "_tower_names") else []
        relation_names = tt.relation_names if hasattr(tt, "relation_names") else relation_names
        base_kinds = {
            name: getattr(getattr(tt, name), "output_kind", "binary_logit") for name in names
        }
    else:
        names = []
        base_kinds = (
            {tower.name: tower.output_kind for tower in task_config.towers}
            if task_config is not None
            else {}
        )
    label_col_map = (params or {}).get("label_col_map", {n: n for n in names} if names else {})
    return {
        "task_names": names,
        "label_col_map": label_col_map,
        "output_kinds": _output_kinds(names, relation_names, base_kinds),
    }


def _build_unimixer(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> UniMixerModel:
    if tokenizer is None:
        raise ValueError("UniMixer requires external FeatureTokenizer")
    output_contract = _parse_output_contract(params)
    tc = _parse_task_config(params.get("task_config"))
    if tc is None and output_contract is None:
        raise ValueError("UniMixer requires task_config or output_contract")
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
        output_contract=output_contract,
    )


def _build_token_mixer_large(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> TokenMixerLargeModel:
    if tokenizer is None:
        raise ValueError("TokenMixerLarge requires external FeatureTokenizer")
    output_contract = _parse_output_contract(params)
    tc = _parse_task_config(params.get("task_config"))
    if tc is None and output_contract is None:
        raise ValueError("TokenMixerLarge requires task_config or output_contract")
    return TokenMixerLargeModel(
        tokenizer=tokenizer,
        token_dim=params.get("token_dim", 64),
        num_tokens=params.get("num_tokens", 8),
        num_blocks=params.get("num_blocks", 2),
        num_heads=params.get("num_heads", 8),
        hidden_factor=params.get("hidden_factor", 1.0),
        task_config=tc,
        down_init_scale=params.get("down_init_scale", 0.01),
        output_contract=output_contract,
    )


def _build_pepnet(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> PEPNet:
    output_contract = _parse_output_contract(params)
    task_config = _parse_task_config(params.get("task_config"))
    if task_config is None and output_contract is None:
        raise ValueError("PEPNet requires task_config or output_contract")
    return PEPNet(
        features,
        prior_dim=params.get("prior_dim", 16),
        deep_hidden_dims=params.get("deep_hidden_dims", []),
        shared_bottom_dims=params.get("shared_bottom_dims", []),
        task_config=task_config,
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
        output_contract=output_contract,
        ep_prior_features=params.get("ep_prior_features"),
        pp_prior_features=params.get("pp_prior_features"),
    )


def _build_rankmixer(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> RankMixerModel:
    if tokenizer is None:
        raise ValueError("RankMixer requires external FeatureTokenizer")
    output_contract = _parse_output_contract(params)
    tc = _parse_task_config(params.get("task_config"))
    if tc is None and output_contract is None:
        raise ValueError("RankMixer requires task_config or output_contract")
    num_tokens = params.get("num_tokens", 8)
    return RankMixerModel(
        tokenizer=tokenizer,
        token_dim=params.get("token_dim", 64),
        num_tokens=num_tokens,
        num_blocks=params.get("num_blocks", 2),
        num_heads=params.get("num_heads", num_tokens),
        hidden_factor=params.get("hidden_factor", 1.0),
        task_config=tc,
        output_contract=output_contract,
    )


def _build_fat(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> FATModel:
    output_contract = _parse_output_contract(params)
    task_config = _parse_task_config(params.get("task_config"))
    if task_config is None and output_contract is None:
        raise ValueError("FAT requires task_config or output_contract")
    return FATModel(
        features,
        d=params.get("d", 128),
        d_ff=params.get("d_ff", 512),
        num_layers=params.get("num_layers", 2),
        n_heads=params.get("n_heads", 8),
        M=params.get("M", 64),
        k=params.get("k", 64),
        K=params.get("K", 3),
        dropout=params.get("dropout", 0.0),
        deep_hidden_dims=params.get("deep_hidden_dims", []),
        shared_bottom_dims=params.get("shared_bottom_dims", []),
        task_config=task_config,
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
        output_contract=output_contract,
    )


def _build_onerank(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> OneRankModel:
    output_contract = _parse_output_contract(params)
    return OneRankModel(
        features,
        d=params.get("d", 128),
        d_ff=params.get("d_ff", 512),
        num_layers=params.get("num_layers", 2),
        n_heads=params.get("n_heads", 8),
        num_tasks=params.get("num_tasks", 3),
        cross_task_mask=params.get("cross_task_mask", "cascade"),
        dropout=params.get("dropout", 0.0),
        pooling_map=params.get("_pooling_map"),
        total_dim=params.get("_total_dim"),
        output_contract=output_contract,
    )


def _build_mixformer(
    features: list[FeatureTuple], tokenizer: nn.Module | None = None, **params: Any
) -> MixFormerModel:
    output_contract = _parse_output_contract(params)
    total_dim = params.get("_total_dim")
    if total_dim is None:
        total_dim = sum(f[2] for f in features)
    return MixFormerModel(
        features,
        d=params.get("d", 386),
        d_ff=params.get("d_ff", 1024),
        num_heads=params.get("num_heads", 16),
        num_layers=params.get("num_layers", 4),
        dropout=params.get("dropout", 0.0),
        pooling_map=params.get("_pooling_map"),
        total_dim=total_dim,
        output_contract=output_contract,
    )


register_model("lr", _spec_pred, _build_lr)
register_model("deepfm", _spec_pred, _build_deepfm)
register_model("mixformer", _spec_pred, _build_mixformer)
register_model("mmoe", _spec_mmoe, _build_mmoe)
register_model("esmm", _spec_esmm, _build_esmm)
register_model("gdcn_esmm", _spec_esmm, _build_gdcn_esmm)
register_model("unimixer", _spec_unimixer, _build_unimixer)
register_model("token_mixer_large", _spec_unimixer, _build_token_mixer_large)
register_model("rankmixer", _spec_unimixer, _build_rankmixer)
register_model("pepnet", _spec_esmm, _build_pepnet)
register_model("fat", _spec_pred, _build_fat)
register_model("onerank", _spec_pred, _build_onerank)
