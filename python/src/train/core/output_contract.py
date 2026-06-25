from __future__ import annotations

"""Output contract v1 schema and semantic validation."""

import json
import math
from dataclasses import dataclass
from typing import Any

NODE_KINDS = {"binary_logit", "probability", "regression", "score"}
TOWER_KINDS = {"binary_logit", "regression", "score"}
ACTIVATIONS = {"relu", "sigmoid", "swish", "gelu", "none"}
RELATION_OPS = {"sigmoid", "multiply", "add", "identity"}
REDUCTIONS = {"mean", "sum"}
MASK_OPS = {"eq", "ne", "gt", "ge", "lt", "le", "is_null", "not_null"}


@dataclass(frozen=True)
class SourceContract:
    name: str
    role: str
    has_default: bool = False


@dataclass(frozen=True)
class TowerNode:
    name: str
    kind: str
    input: str = "shared"
    hidden_dims: tuple[int, ...] = ()
    activation: str = "relu"


@dataclass(frozen=True)
class RelationNode:
    name: str
    op: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class MaskSpec:
    source: str
    op: str
    value: Any = None


@dataclass(frozen=True)
class LossSpec:
    type: str
    reduction: str = "mean"
    epsilon: float | None = None
    pos_weight: float | None = None
    delta: float | None = None
    alpha: float | None = None
    gamma: float | None = None


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    source: str
    label: str
    loss: LossSpec
    weight: float = 1.0
    mask: MaskSpec | None = None


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    label: str
    type: str
    mask: MaskSpec | None = None


@dataclass(frozen=True)
class PublicOutputSpec:
    name: str
    source: str


@dataclass(frozen=True)
class NormalizedOutputContract:
    version: int
    towers: tuple[TowerNode, ...]
    relations: tuple[RelationNode, ...]
    relation_order: tuple[str, ...]
    node_kinds: dict[str, str]
    objectives: tuple[ObjectiveSpec, ...]
    metrics: tuple[MetricSpec, ...]
    outputs: tuple[PublicOutputSpec, ...]

    def canonical_json(self) -> str:
        relations = {relation.name: relation for relation in self.relations}
        data = {
            "graph": {
                "relations": [
                    {
                        "inputs": list(relations[name].inputs),
                        "name": name,
                        "op": relations[name].op,
                    }
                    for name in self.relation_order
                ],
                "towers": [
                    {
                        "activation": tower.activation,
                        "hidden_dims": list(tower.hidden_dims),
                        "input": tower.input,
                        "kind": tower.kind,
                        "name": tower.name,
                    }
                    for tower in sorted(self.towers, key=lambda item: item.name)
                ],
            },
            "metrics": [
                _metric_data(metric) for metric in sorted(self.metrics, key=lambda item: item.name)
            ],
            "objectives": [
                _objective_data(objective)
                for objective in sorted(self.objectives, key=lambda item: item.name)
            ],
            "outputs": [
                {"name": output.name, "source": output.source}
                for output in sorted(self.outputs, key=lambda item: item.name)
            ],
            "version": self.version,
        }
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_output_contract(
    raw: dict[str, Any], sources: list[SourceContract] | None = None
) -> NormalizedOutputContract:
    _keys(
        "output_contract",
        raw,
        {"version", "graph", "objectives", "metrics", "outputs"},
        {"version", "graph", "objectives", "metrics", "outputs"},
    )
    if raw["version"] != 1:
        raise ValueError(f"output_contract.version must be 1, got {raw['version']}")
    graph = _mapping("output_contract.graph", raw["graph"])
    _keys("output_contract.graph", graph, {"towers", "relations"}, {"towers", "relations"})

    towers = tuple(_parse_tower(i, value) for i, value in enumerate(graph["towers"]))
    relations = tuple(_parse_relation(i, value) for i, value in enumerate(graph["relations"]))
    objectives = tuple(_parse_objective(i, value) for i, value in enumerate(raw["objectives"]))
    metrics = tuple(_parse_metric(i, value) for i, value in enumerate(raw["metrics"]))
    outputs = tuple(_parse_output(i, value) for i, value in enumerate(raw["outputs"]))

    node_kinds, relation_order = _infer_graph(towers, relations)
    _unique("objective", [item.name for item in objectives])
    _unique("metric", [item.name for item in metrics])
    _unique("public output", [item.name for item in outputs])
    _validate_references(node_kinds, relations, objectives, metrics, outputs)
    _validate_objectives(node_kinds, objectives)
    _validate_metrics(node_kinds, metrics)
    if sources is not None:
        _validate_sources(sources, objectives, metrics)

    return NormalizedOutputContract(
        version=1,
        towers=towers,
        relations=relations,
        relation_order=relation_order,
        node_kinds=node_kinds,
        objectives=objectives,
        metrics=metrics,
        outputs=outputs,
    )


def _parse_tower(index: int, raw: Any) -> TowerNode:
    value = _mapping(f"towers[{index}]", raw)
    _keys(
        f"towers[{index}]",
        value,
        {"name", "input", "kind", "hidden_dims", "activation"},
        {"name", "kind"},
    )
    kind = str(value["kind"])
    if kind not in TOWER_KINDS:
        raise ValueError(
            f"tower '{value['name']}' kind '{kind}' is invalid; probability towers are forbidden"
        )
    activation = str(value.get("activation", "relu"))
    if activation not in ACTIVATIONS:
        raise ValueError(f"tower '{value['name']}' has unsupported activation '{activation}'")
    dims = tuple(value.get("hidden_dims", []))
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in dims):
        raise ValueError(f"tower '{value['name']}' hidden_dims must contain positive integers")
    return TowerNode(str(value["name"]), kind, str(value.get("input", "shared")), dims, activation)


def _parse_relation(index: int, raw: Any) -> RelationNode:
    value = _mapping(f"relations[{index}]", raw)
    _keys(f"relations[{index}]", value, {"name", "op", "inputs"}, {"name", "op", "inputs"})
    op = str(value["op"])
    if op not in RELATION_OPS:
        raise ValueError(f"relation '{value['name']}' has unsupported op '{op}'")
    return RelationNode(str(value["name"]), op, tuple(str(v) for v in value["inputs"]))


def _parse_mask(raw: Any, context: str) -> MaskSpec | None:
    if raw is None:
        return None
    value = _mapping(context, raw)
    _keys(context, value, {"source", "op", "value"}, {"source", "op"})
    op = str(value["op"])
    if op not in MASK_OPS:
        raise ValueError(f"{context}.op '{op}' is unsupported")
    if op not in {"is_null", "not_null"} and "value" not in value:
        raise ValueError(f"{context}.value is required for op '{op}'")
    if op in {"is_null", "not_null"} and "value" in value:
        raise ValueError(f"{context}.value is not allowed for op '{op}'")
    return MaskSpec(str(value["source"]), op, value.get("value"))


def _parse_loss(raw: Any, context: str) -> LossSpec:
    value = _mapping(context, raw)
    _keys(
        context,
        value,
        {"type", "reduction", "epsilon", "pos_weight", "delta", "alpha", "gamma"},
        {"type"},
    )
    reduction = str(value.get("reduction", "mean"))
    if reduction not in REDUCTIONS:
        raise ValueError(f"{context}.reduction must be mean or sum")
    loss_type = str(value["type"])
    allowed_params = {
        "binary_cross_entropy_with_logits": {"pos_weight"},
        "binary_cross_entropy": {"epsilon"},
        "focal_binary_cross_entropy": {"epsilon", "alpha", "gamma"},
        "focal_binary_cross_entropy_with_logits": {"alpha", "gamma"},
        "mse": set(),
        "mae": set(),
        "huber": {"delta"},
        "weighted_bce_stay": set(),
    }
    if loss_type not in allowed_params:
        raise ValueError(f"{context}.type '{loss_type}' is unsupported")
    supplied = {key for key in ("epsilon", "pos_weight", "delta", "alpha", "gamma") if key in value}
    invalid = supplied - allowed_params[loss_type]
    if invalid:
        raise ValueError(f"{context} has parameters not valid for {loss_type}: {sorted(invalid)}")
    epsilon = value.get(
        "epsilon",
        1e-7
        if loss_type in {"binary_cross_entropy", "focal_binary_cross_entropy"}
        else None,
    )
    if epsilon is not None and not (0.0 < float(epsilon) < 0.5):
        raise ValueError(f"{context}.epsilon must be between 0 and 0.5")
    weight = value.get("pos_weight")
    if weight is not None and (not math.isfinite(float(weight)) or float(weight) <= 0):
        raise ValueError(f"{context}.pos_weight must be > 0")
    delta = value.get("delta", 1.0 if loss_type == "huber" else None)
    if delta is not None and (not math.isfinite(float(delta)) or float(delta) <= 0):
        raise ValueError(f"{context}.delta must be > 0")
    alpha = value.get("alpha")
    if alpha is not None and (not math.isfinite(float(alpha)) or not (0.0 <= float(alpha) <= 1.0)):
        raise ValueError(f"{context}.alpha must be between 0 and 1")
    gamma = value.get(
        "gamma",
        2.0
        if loss_type
        in {"focal_binary_cross_entropy", "focal_binary_cross_entropy_with_logits"}
        else None,
    )
    if gamma is not None and (not math.isfinite(float(gamma)) or float(gamma) < 0):
        raise ValueError(f"{context}.gamma must be >= 0")
    return LossSpec(
        loss_type,
        reduction,
        None if epsilon is None else float(epsilon),
        None if weight is None else float(weight),
        None if delta is None else float(delta),
        None if alpha is None else float(alpha),
        None if gamma is None else float(gamma),
    )


def _parse_objective(index: int, raw: Any) -> ObjectiveSpec:
    context = f"objectives[{index}]"
    value = _mapping(context, raw)
    _keys(
        context,
        value,
        {"name", "source", "label", "loss", "weight", "mask"},
        {"name", "source", "label", "loss"},
    )
    weight = float(value.get("weight", 1.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{context}.weight must be finite and non-negative")
    return ObjectiveSpec(
        str(value["name"]),
        str(value["source"]),
        str(value["label"]),
        _parse_loss(value["loss"], f"{context}.loss"),
        weight,
        _parse_mask(value.get("mask"), f"{context}.mask"),
    )


def _parse_metric(index: int, raw: Any) -> MetricSpec:
    context = f"metrics[{index}]"
    value = _mapping(context, raw)
    _keys(
        context,
        value,
        {"name", "source", "label", "type", "mask"},
        {"name", "source", "label", "type"},
    )
    metric_type = str(value["type"])
    if metric_type not in {"auc", "prauc", "logloss", "mae", "mse"}:
        raise ValueError(f"{context}.type '{metric_type}' is unsupported")
    return MetricSpec(
        str(value["name"]),
        str(value["source"]),
        str(value["label"]),
        metric_type,
        _parse_mask(value.get("mask"), f"{context}.mask"),
    )


def _parse_output(index: int, raw: Any) -> PublicOutputSpec:
    context = f"outputs[{index}]"
    value = _mapping(context, raw)
    _keys(context, value, {"name", "source"}, {"name", "source"})
    return PublicOutputSpec(str(value["name"]), str(value["source"]))


def _infer_graph(
    towers: tuple[TowerNode, ...], relations: tuple[RelationNode, ...]
) -> tuple[dict[str, str], tuple[str, ...]]:
    names = [tower.name for tower in towers] + [relation.name for relation in relations]
    _unique("graph node", names)
    kinds = {tower.name: tower.kind for tower in towers}
    pending = {relation.name: relation for relation in relations}
    order: list[str] = []
    while pending:
        ready = sorted(
            name
            for name, relation in pending.items()
            if all(source in kinds for source in relation.inputs)
        )
        if not ready:
            unknown = sorted(
                {
                    source
                    for relation in pending.values()
                    for source in relation.inputs
                    if source not in kinds and source not in pending
                }
            )
            if unknown:
                raise ValueError(f"relation references unknown node(s): {unknown}")
            raise ValueError(f"output graph contains a cycle: {sorted(pending)}")
        for name in ready:
            relation = pending.pop(name)
            kinds[name] = _relation_kind(relation, kinds)
            order.append(name)
    return kinds, tuple(order)


def _relation_kind(relation: RelationNode, kinds: dict[str, str]) -> str:
    input_kinds = [kinds[name] for name in relation.inputs]
    if relation.op == "sigmoid":
        _arity(relation, 1)
        if input_kinds != ["binary_logit"]:
            raise ValueError(f"relation '{relation.name}' sigmoid requires binary_logit")
        return "probability"
    if relation.op == "multiply":
        if len(input_kinds) < 2 or any(kind != "probability" for kind in input_kinds):
            raise ValueError(
                f"relation '{relation.name}' multiply requires at least two probability inputs"
            )
        return "probability"
    if relation.op == "add":
        if len(input_kinds) < 2 or any(kind != "regression" for kind in input_kinds):
            raise ValueError(
                f"relation '{relation.name}' add requires at least two regression inputs"
            )
        return "regression"
    _arity(relation, 1)
    return input_kinds[0]


def _validate_references(
    kinds: dict[str, str],
    relations: tuple[RelationNode, ...],
    objectives: tuple[ObjectiveSpec, ...],
    metrics: tuple[MetricSpec, ...],
    outputs: tuple[PublicOutputSpec, ...],
) -> None:
    consumed = {source for relation in relations for source in relation.inputs}
    for group in (objectives, metrics, outputs):
        for item in group:
            if item.source not in kinds:
                raise ValueError(f"'{item.source}' references unknown output graph node")
            consumed.add(item.source)
    unused = sorted(set(kinds) - consumed)
    if unused:
        raise ValueError(f"output graph has unused node(s): {unused}")


def _validate_objectives(kinds: dict[str, str], objectives: tuple[ObjectiveSpec, ...]) -> None:
    expected = {
        "binary_cross_entropy_with_logits": {"binary_logit"},
        "focal_binary_cross_entropy": {"probability"},
        "focal_binary_cross_entropy_with_logits": {"binary_logit"},
        "binary_cross_entropy": {"probability"},
        "mse": {"regression", "score"},
        "mae": {"regression", "score"},
        "huber": {"regression", "score"},
        "weighted_bce_stay": {"binary_logit"},
    }
    for item in objectives:
        if kinds[item.source] not in expected[item.loss.type]:
            raise ValueError(
                f"objective '{item.name}' loss {item.loss.type} cannot consume {kinds[item.source]}"
            )


def _validate_metrics(kinds: dict[str, str], metrics: tuple[MetricSpec, ...]) -> None:
    expected = {
        "auc": {"binary_logit", "probability"},
        "prauc": {"binary_logit", "probability"},
        "logloss": {"probability"},
        "mae": {"regression", "score"},
        "mse": {"regression", "score"},
    }
    for item in metrics:
        if kinds[item.source] not in expected[item.type]:
            raise ValueError(
                f"metric '{item.name}' type {item.type} cannot consume {kinds[item.source]}"
            )


def _validate_sources(
    sources: list[SourceContract],
    objectives: tuple[ObjectiveSpec, ...],
    metrics: tuple[MetricSpec, ...],
) -> None:
    catalog = {source.name: source for source in sources}
    for source in sources:
        if source.role == "label" and source.has_default:
            raise ValueError(f"label source '{source.name}' must not define a default")
    for item in (*objectives, *metrics):
        label = catalog.get(item.label)
        if label is None or label.role != "label":
            raise ValueError(f"'{item.label}' must reference a label source")
        if item.mask is not None and item.mask.source not in catalog:
            raise ValueError(f"mask source '{item.mask.source}' does not exist")


def _arity(relation: RelationNode, count: int) -> None:
    if len(relation.inputs) != count:
        raise ValueError(f"relation '{relation.name}' {relation.op} requires {count} input(s)")


def _mask_data(mask: MaskSpec | None) -> dict[str, Any] | None:
    if mask is None:
        return None
    return {"op": mask.op, "source": mask.source, "value": mask.value}


def _objective_data(objective: ObjectiveSpec) -> dict[str, Any]:
    return {
        "label": objective.label,
        "loss": {
            "alpha": _canonical_float(objective.loss.alpha),
            "delta": _canonical_float(objective.loss.delta),
            "epsilon": _canonical_float(objective.loss.epsilon),
            "gamma": _canonical_float(objective.loss.gamma),
            "pos_weight": _canonical_float(objective.loss.pos_weight),
            "reduction": objective.loss.reduction,
            "type": objective.loss.type,
        },
        "mask": _mask_data(objective.mask),
        "name": objective.name,
        "source": objective.source,
        "weight": _canonical_float(objective.weight),
    }


def _metric_data(metric: MetricSpec) -> dict[str, Any]:
    return {
        "label": metric.label,
        "mask": _mask_data(metric.mask),
        "name": metric.name,
        "source": metric.source,
        "type": metric.type,
    }


def _canonical_float(value: float | None) -> str | None:
    if value is None:
        return None
    mantissa, exponent = format(value, ".17e").split("e")
    return f"{mantissa}e{int(exponent)}"


def _unique(context: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {context} name(s): {duplicates}")


def _mapping(context: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{context} must be a mapping")
    return raw


def _keys(context: str, raw: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    missing = sorted(required - set(raw))
    if unknown:
        raise ValueError(f"{context} has unknown field(s): {unknown}")
    if missing:
        raise ValueError(f"{context} missing required field(s): {missing}")
