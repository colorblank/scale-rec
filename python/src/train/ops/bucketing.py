from __future__ import annotations

"""分桶算子：连续值到桶索引的映射。"""
from typing import Any


class Bucketing:
    def __init__(self, boundaries: list[float]):
        self.boundaries = sorted(boundaries)

    def process(self, inputs: list[Any]) -> int:
        val = float(inputs[0])
        bucket = 0
        for b in self.boundaries:
            if val < b:
                break
            bucket += 1
        return bucket
