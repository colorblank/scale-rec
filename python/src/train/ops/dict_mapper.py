from __future__ import annotations

"""字典映射算子：字符串/整数到索引的映射。"""
from typing import Any


class DictMapper:
    """String/int to index lookup."""

    def __init__(self, mapping: dict[str, int], default_idx: int = 0):
        """Initialize with mapping table and fallback index."""
        self.mapping = mapping
        self.default_idx = default_idx

    def process(self, inputs: list[Any]) -> int | list[int]:
        """Map string key to integer index; list input -> list output, single -> single."""
        val = inputs[0]
        if isinstance(val, list):
            return [
                self.mapping.get(str(v), self.default_idx)
                if isinstance(v, (str, int, float))
                else self.default_idx
                for v in val
            ]
        if isinstance(val, (int, float)):
            val = str(val)
        return self.mapping.get(val, self.default_idx) if isinstance(val, str) else self.default_idx

    def process_batch(self, inputs: list[Any]) -> list:
        """Batch map: N values -> N results. Single call for entire column."""
        vals = inputs[0]
        if not vals:
            return []
        first = vals[0]
        if isinstance(first, list):
            return [
                [
                    self.mapping.get(str(v), self.default_idx)
                    if isinstance(v, (str, int, float))
                    else self.default_idx
                    for v in row
                ]
                for row in vals
            ]
        return [
            self.mapping.get(str(v), self.default_idx)
            if isinstance(v, (str, int, float))
            else self.default_idx
            for v in vals
        ]
