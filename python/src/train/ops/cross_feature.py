from __future__ import annotations

"""特征交叉算子：内积或笛卡尔积。"""
from typing import Any

from ..core.config import CrossType
from . import register_op


@register_op("CrossFeature")
class CrossFeature:
    """Feature cross: inner product or cartesian product."""

    def __init__(self, cross_type: CrossType) -> None:
        if isinstance(cross_type, str):
            cross_type = CrossType(cross_type)
        self.cross_type = cross_type

    @classmethod
    def from_config(cls, params: dict) -> CrossFeature:
        return cls(CrossType(params.get("cross_type", "cartesian")))

    def process(self, inputs: list[Any]) -> float | list[str]:
        """Compute cross: dot product or all-pairs concat."""
        if len(inputs) != 2:
            raise ValueError(f"CrossFeature expects exactly 2 inputs, got {len(inputs)}")
        a, b = inputs[0], inputs[1]
        if self.cross_type is CrossType.INNER_PRODUCT:
            va = [float(x) for x in a]
            vb = [float(x) for x in b]
            return sum(x * y for x, y in zip(va, vb, strict=False))
        sa = [str(x) for x in a]
        sb = [str(x) for x in b]
        return [f"{x}_{y}" for x in sa for y in sb]
