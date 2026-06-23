from __future__ import annotations

"""log1p 数值算子：计算 ln(1 + x)。"""

import math
from typing import Any

from . import register_op


@register_op("Log1p")
class Log1p:
    """Single-input numeric log1p operator."""

    @classmethod
    def from_config(cls, params: dict) -> Log1p:
        return cls()

    def process(self, inputs: list[Any]) -> float:
        value = float(inputs[0])
        if value <= -1.0:
            raise ValueError("Log1p: input must be greater than -1")
        return float(math.log1p(value))

    def process_batch(self, inputs: list[Any]) -> list[float]:
        values = inputs[0]
        result = []
        for raw in values:
            value = float(raw)
            if value <= -1.0:
                raise ValueError("Log1p: input must be greater than -1")
            result.append(float(math.log1p(value)))
        return result
