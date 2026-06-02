from __future__ import annotations

"""Task contract shared by model output, labels, loss, and metrics."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    loss: str = "bce"
    weight: float = 1.0
    mask: Optional[str] = None
    pos_weight: Optional[float] = None
    metrics: tuple[str, ...] = field(default_factory=tuple)


def parse_task_specs(raw: Optional[list[dict[str, Any]]]) -> list[TaskSpec]:
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
        if loss not in {"bce", "weighted_bce_stay"}:
            raise ValueError(f"Unsupported loss for task '{name}': {loss}")
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
            )
        )
    return specs


def legacy_task_specs(
    task_names: list[str],
    label_map: dict[str, str],
    task_weights: Optional[dict[str, float]] = None,
) -> list[TaskSpec]:
    weights = task_weights or {}
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
            )
        )
    return specs


def task_names(specs: list[TaskSpec]) -> list[str]:
    return [spec.name for spec in specs]


def label_map(specs: list[TaskSpec]) -> dict[str, str]:
    return {spec.name: spec.label for spec in specs}
