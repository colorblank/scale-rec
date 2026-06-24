from __future__ import annotations

"""Structured model outputs with explicit tensor semantics."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

OutputKind = str
BINARY_LOGIT: OutputKind = "binary_logit"
PROBABILITY: OutputKind = "probability"
REGRESSION: OutputKind = "regression"
SCORE: OutputKind = "score"


@dataclass(frozen=True)
class OutputTensor:
    tensor: torch.Tensor
    kind: OutputKind


class ModelOutput:
    """Model output container that binds every tensor to an output kind."""

    def __init__(self, values: Mapping[str, OutputTensor] | None = None) -> None:
        self._values: dict[str, OutputTensor] = dict(values or {})

    @classmethod
    def from_tensors(
        cls, tensors: Mapping[str, torch.Tensor], output_kinds: Mapping[str, OutputKind]
    ) -> ModelOutput:
        return cls(
            {
                name: OutputTensor(tensor, output_kinds.get(name, BINARY_LOGIT))
                for name, tensor in tensors.items()
            }
        )

    @classmethod
    def binary_logits(cls, tensors: Mapping[str, torch.Tensor]) -> ModelOutput:
        return cls.from_tensors(tensors, dict.fromkeys(tensors, BINARY_LOGIT))

    def insert(self, name: str, tensor: torch.Tensor, kind: OutputKind) -> None:
        self._values[name] = OutputTensor(tensor, kind)

    def insert_binary_logit(self, name: str, tensor: torch.Tensor) -> None:
        self.insert(name, tensor, BINARY_LOGIT)

    def insert_probability(self, name: str, tensor: torch.Tensor) -> None:
        self.insert(name, tensor, PROBABILITY)

    def insert_regression(self, name: str, tensor: torch.Tensor) -> None:
        self.insert(name, tensor, REGRESSION)

    def insert_score(self, name: str, tensor: torch.Tensor) -> None:
        self.insert(name, tensor, SCORE)

    def get(self, name: str) -> OutputTensor | None:
        return self._values.get(name)

    def tensor(self, name: str) -> torch.Tensor:
        return self._values[name].tensor

    def kind(self, name: str) -> OutputKind:
        return self._values[name].kind

    def names(self) -> list[str]:
        return list(self._values)

    def items(self):
        return self._values.items()

    def tensor_items(self):
        for name, output in self._values.items():
            yield name, output.tensor

    def with_kinds(self, output_kinds: Mapping[str, OutputKind]) -> ModelOutput:
        if not output_kinds:
            return self
        return ModelOutput(
            {
                name: OutputTensor(output.tensor, output_kinds.get(name, output.kind))
                for name, output in self._values.items()
            }
        )

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class ModelExecution:
    """Complete output graph execution and its public projection."""

    nodes: ModelOutput
    outputs: ModelOutput


def ensure_model_output(
    outputs: ModelOutput | Mapping[str, torch.Tensor],
    output_kinds: Mapping[str, OutputKind] | None = None,
) -> ModelOutput:
    if isinstance(outputs, ModelOutput):
        return outputs.with_kinds(output_kinds or {})
    return ModelOutput.from_tensors(outputs, output_kinds or {})
