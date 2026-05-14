from __future__ import annotations

"""字典映射算子：字符串/整数到索引的映射。"""
from typing import Any


class DictMapper:
    """String/int to index lookup."""

    def __init__(self, mapping: dict[str, int], default_idx: int = 0):
        """Initialize with mapping table and fallback index."""
        self.mapping = mapping
        self.default_idx = default_idx

    def process(self, inputs: list[Any]) -> int:
        """Map string key to integer index; returns default_idx for unknown keys."""
        val = inputs[0]
        if isinstance(val, (int, float)):
            val = str(val)
        return self.mapping.get(val, self.default_idx) if isinstance(val, str) else self.default_idx
