from __future__ import annotations

"""Task contract shared by model output, labels, loss, and metrics."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    loss: str = "bce"
    weight: float = 1.0
    mask: str | None = None
    pos_weight: float | None = None
    metrics: tuple[str, ...] = field(default_factory=tuple)
    output_kind: str = "binary_logit"


@dataclass(frozen=True)
class TaskContract:
    specs: tuple[TaskSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_specs(cls, specs: list[TaskSpec] | tuple[TaskSpec, ...] | None) -> TaskContract:
        return cls(tuple(specs or ()))

    @classmethod
    def from_raw(cls, raw: list[dict[str, Any]] | None) -> TaskContract:
        return cls.from_specs(parse_task_specs(raw))

    @property
    def task_names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def label_col_map(self) -> dict[str, str]:
        return {spec.name: spec.label for spec in self.specs}

    @property
    def output_kinds(self) -> dict[str, str]:
        return {spec.name: spec.output_kind for spec in self.specs}

    @property
    def task_metrics(self) -> dict[str, list[str]]:
        return {spec.name: list(spec.metrics) for spec in self.specs}

    def to_manifest(self) -> list[dict[str, Any]]:
        return [asdict(spec) | {"metrics": list(spec.metrics)} for spec in self.specs]


def parse_task_specs(raw: list[dict[str, Any]] | None) -> list[TaskSpec]:
    if not raw:
        return []
    specs = []
    seen = set()
    for item in raw:
        name = str(item["name"])
        if name in seen:
            raise ValueError(f"Duplicate task spec: {name}")
        seen.add(name)
        loss = str(item.get("loss", "bce"))
        if loss not in {"bce", "weighted_bce_stay", "mse", "mae", "huber"}:
            raise ValueError(f"Unsupported loss for task '{name}': {loss}")
        output_kind = str(item.get("output_kind", item.get("output", _default_output_kind(loss))))
        if output_kind not in {"binary_logit", "probability", "regression", "score"}:
            raise ValueError(f"Unsupported output_kind for task '{name}': {output_kind}")
        metrics = item.get("metrics", ())
        if isinstance(metrics, str):
            metrics = tuple(x.strip() for x in metrics.split(",") if x.strip())
        specs.append(
            TaskSpec(
                name=name,
                label=str(item.get("label", name)),
                loss=loss,
                weight=float(item.get("weight", 1.0)),
                mask=item.get("mask"),
                pos_weight=None if item.get("pos_weight") is None else float(item["pos_weight"]),
                metrics=tuple(metrics),
                output_kind=output_kind,
            )
        )
    return specs


def legacy_task_specs(
    task_names: list[str],
    label_map: dict[str, str],
    task_weights: dict[str, float] | None = None,
    default_metrics: list[str] | None = None,
) -> list[TaskSpec]:
    weights = task_weights or {}
    metrics = tuple(default_metrics or ())
    specs = []
    for name in task_names:
        if name.startswith("ct"):
            continue
        specs.append(
            TaskSpec(
                name=name,
                label=label_map.get(name, name),
                loss="weighted_bce_stay" if name == "stay" else "bce",
                weight=float(weights.get(name, 1.0)),
                metrics=metrics,
                output_kind="binary_logit",
            )
        )
    return specs


def task_names(specs: list[TaskSpec]) -> list[str]:
    return TaskContract.from_specs(specs).task_names


def label_map(specs: list[TaskSpec]) -> dict[str, str]:
    return TaskContract.from_specs(specs).label_col_map


def output_kinds(specs: list[TaskSpec]) -> dict[str, str]:
    return TaskContract.from_specs(specs).output_kinds


def task_specs_to_manifest(
    specs: list[TaskSpec] | tuple[TaskSpec, ...] | None,
) -> list[dict[str, Any]]:
    return TaskContract.from_specs(specs).to_manifest()


def _default_output_kind(loss: str) -> str:
    if loss in {"mse", "mae", "huber"}:
        return "regression"
    return "binary_logit"
