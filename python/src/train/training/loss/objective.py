from __future__ import annotations

"""Contract-driven objective evaluation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.model_output import ModelExecution, ModelOutput
from ...core.output_contract import LossSpec, MaskSpec, NormalizedOutputContract, ObjectiveSpec


@dataclass(frozen=True)
class ObjectiveResult:
    """Aggregated objective value plus per-objective diagnostics."""

    total: torch.Tensor | None
    losses: dict[str, torch.Tensor]
    sample_counts: dict[str, int]


class ObjectiveEngine(nn.Module):
    """Compute configured objectives from internal output graph nodes."""

    def __init__(self, contract: NormalizedOutputContract) -> None:
        super().__init__()
        self.objectives = contract.objectives
        self.node_kinds = contract.node_kinds

    def forward(
        self,
        execution: ModelExecution | ModelOutput,
        batch_values: Mapping[str, Sequence[Any] | torch.Tensor | np.ndarray],
    ) -> ObjectiveResult:
        nodes = execution.nodes if isinstance(execution, ModelExecution) else execution
        total: torch.Tensor | None = None
        losses: dict[str, torch.Tensor] = {}
        sample_counts: dict[str, int] = {}
        label_means: dict[str, float] = {}

        for objective in self.objectives:
            output = nodes.get(objective.source)
            if output is None:
                raise ValueError(
                    f"objective '{objective.name}' source '{objective.source}' is missing"
                )
            expected_kind = self.node_kinds[objective.source]
            if output.kind != expected_kind:
                raise ValueError(
                    f"objective '{objective.name}' source '{objective.source}' expected "
                    f"{expected_kind}, got {output.kind}"
                )
            raw_labels = _column(batch_values, objective.label)
            if len(raw_labels) != output.tensor.shape[0]:
                raise ValueError(
                    f"objective '{objective.name}' label batch size {len(raw_labels)} "
                    f"does not match prediction batch size {output.tensor.shape[0]}"
                )
            valid = np.array([not _is_null(value) for value in raw_labels], dtype=bool)
            if objective.mask is not None:
                valid &= evaluate_mask(objective.mask, batch_values, len(raw_labels))
            count = int(valid.sum())
            if count == 0:
                continue

            prediction = output.tensor[torch.from_numpy(valid).to(output.tensor.device)]
            target = torch.tensor(
                [float(value) for value, keep in zip(raw_labels, valid, strict=True) if keep],
                dtype=prediction.dtype,
                device=prediction.device,
            ).reshape_as(prediction)
            raw_loss = _objective_loss(objective, prediction, target)
            losses[objective.name] = raw_loss
            sample_counts[objective.name] = count
            label_means[objective.name] = float(target.detach().mean().item())
            weighted = raw_loss * objective.weight
            total = weighted if total is None else total + weighted

        self._last_losses = {name: float(value.detach().item()) for name, value in losses.items()}
        self._last_sample_counts = sample_counts
        self._last_label_means = label_means
        return ObjectiveResult(total=total, losses=losses, sample_counts=sample_counts)

    def last_losses(self) -> dict[str, float]:
        return getattr(self, "_last_losses", {})

    def last_pos_rates(self) -> dict[str, float]:
        return getattr(self, "_last_label_means", {})

    def task_weights_info(self) -> dict[str, float]:
        return {objective.name: objective.weight for objective in self.objectives}


def _objective_loss(
    objective: ObjectiveSpec, prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    loss = objective.loss
    if loss.type == "binary_cross_entropy_with_logits":
        pos_weight = (
            None
            if loss.pos_weight is None
            else torch.tensor(loss.pos_weight, dtype=prediction.dtype, device=prediction.device)
        )
        return F.binary_cross_entropy_with_logits(
            prediction, target, pos_weight=pos_weight, reduction=loss.reduction
        )
    if loss.type == "binary_cross_entropy":
        epsilon = loss.epsilon if loss.epsilon is not None else 1e-7
        return F.binary_cross_entropy(
            prediction.clamp(epsilon, 1.0 - epsilon),
            target,
            reduction=loss.reduction,
        )
    if loss.type == "focal_binary_cross_entropy":
        epsilon = loss.epsilon if loss.epsilon is not None else 1e-7
        probability = prediction.clamp(epsilon, 1.0 - epsilon)
        values = F.binary_cross_entropy(probability, target, reduction="none")
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        gamma = 2.0 if loss.gamma is None else loss.gamma
        values = values * torch.pow(1.0 - pt, gamma)
        if loss.alpha is not None:
            alpha_t = loss.alpha * target + (1.0 - loss.alpha) * (1.0 - target)
            values = values * alpha_t
        return values.mean() if loss.reduction == "mean" else values.sum()
    if loss.type == "focal_binary_cross_entropy_with_logits":
        return _focal_bce_with_logits(
            prediction,
            target,
            loss,
        )
    if loss.type == "mse":
        return F.mse_loss(prediction, target, reduction=loss.reduction)
    if loss.type == "mae":
        return F.l1_loss(prediction, target, reduction=loss.reduction)
    if loss.type == "huber":
        return F.huber_loss(
            prediction,
            target,
            reduction=loss.reduction,
            delta=loss.delta if loss.delta is not None else 1.0,
        )
    if loss.type == "weighted_bce_stay":
        return _weighted_bce_stay(prediction, target, loss)
    raise ValueError(f"unsupported objective loss '{loss.type}'")


def _weighted_bce_stay(
    logits: torch.Tensor, stay_times: torch.Tensor, loss: LossSpec
) -> torch.Tensor:
    positive = stay_times / (1.0 + stay_times)
    negative = 1.0 / (1.0 + stay_times)
    values = -(positive * -F.softplus(-logits) + negative * -F.softplus(logits))
    return values.mean() if loss.reduction == "mean" else values.sum()


def _focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss: LossSpec,
) -> torch.Tensor:
    values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    gamma = 2.0 if loss.gamma is None else loss.gamma
    values = values * torch.pow(1.0 - pt, gamma)
    if loss.alpha is not None:
        alpha_t = loss.alpha * target + (1.0 - loss.alpha) * (1.0 - target)
        values = values * alpha_t
    return values.mean() if loss.reduction == "mean" else values.sum()


def _column(
    batch_values: Mapping[str, Sequence[Any] | torch.Tensor | np.ndarray], name: str
) -> list[Any]:
    if name not in batch_values:
        raise ValueError(f"batch column '{name}' is missing")
    values = batch_values[name]
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().reshape(-1).tolist()
    if isinstance(values, np.ndarray):
        return values.reshape(-1).tolist()
    return list(values)


def evaluate_mask(
    mask: MaskSpec,
    batch_values: Mapping[str, Sequence[Any] | torch.Tensor | np.ndarray],
    batch_size: int,
) -> np.ndarray:
    values = _column(batch_values, mask.source)
    if len(values) != batch_size:
        raise ValueError(
            f"mask source '{mask.source}' batch size {len(values)} does not match {batch_size}"
        )
    if mask.op == "is_null":
        return np.array([_is_null(value) for value in values], dtype=bool)
    if mask.op == "not_null":
        return np.array([not _is_null(value) for value in values], dtype=bool)

    result = np.zeros(batch_size, dtype=bool)
    for index, value in enumerate(values):
        if _is_null(value):
            continue
        if mask.op == "eq":
            result[index] = value == mask.value
        elif mask.op == "ne":
            result[index] = value != mask.value
        elif mask.op == "gt":
            result[index] = value > mask.value
        elif mask.op == "ge":
            result[index] = value >= mask.value
        elif mask.op == "lt":
            result[index] = value < mask.value
        elif mask.op == "le":
            result[index] = value <= mask.value
        else:
            raise ValueError(f"unsupported mask op '{mask.op}'")
    return result


def _is_null(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))
