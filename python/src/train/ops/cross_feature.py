from __future__ import annotations

"""特征交叉算子：内积或笛卡尔积。"""
from typing import Any


class CrossFeature:
    """Feature cross: inner product or cartesian product."""

    def __init__(self, cross_type: str) -> None:
        """Initialize with "inner_product" or "cartesian" mode."""
        self.cross_type = cross_type

    def process(self, inputs: list[Any]) -> float | list[str]:
        """Compute cross: dot product (inner_product) or all-pairs concat (cartesian)."""
        if len(inputs) != 2:
            raise ValueError(f"CrossFeature expects exactly 2 inputs, got {len(inputs)}")
        a, b = inputs[0], inputs[1]
        if self.cross_type == "inner_product":
            va = [float(x) for x in a]
            vb = [float(x) for x in b]
            return sum(x * y for x, y in zip(va, vb))
        sa = [str(x) for x in a]
        sb = [str(x) for x in b]
        return [f"{x}_{y}" for x in sa for y in sb]
