from __future__ import annotations

"""分桶算子：连续值到桶索引的映射。"""
from typing import Any

from . import register_op


@register_op("Bucketing")
class Bucketing:
    """Continuous value to bucket index."""

    def __init__(self, boundaries: list[float]) -> None:
        """Initialize with sorted boundary thresholds."""
        self.boundaries = sorted(boundaries)

    @classmethod
    def from_config(cls, params: dict) -> "Bucketing":
        return cls([float(x) for x in params.get("boundaries", [])])

    def process(self, inputs: list[Any]) -> int:
        """Map float value to bucket index (count of boundaries <= value)."""
        val = float(inputs[0])
        bucket = 0
        for b in self.boundaries:
            if val < b:
                break
            bucket += 1
        return bucket

    def process_batch(self, inputs: list[Any]) -> list[int]:
        """Batch bucketing: N values -> N bucket indices."""
        vals = inputs[0]
        if not vals:
            return []
        result = []
        boundaries = self.boundaries
        for val in vals:
            v = float(val)
            bucket = 0
            for b in boundaries:
                if v < b:
                    break
                bucket += 1
            result.append(bucket)
        return result
