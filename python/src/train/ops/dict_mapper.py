from __future__ import annotations

"""字典映射算子：字符串/整数到索引的映射。"""
from typing import Any, Union

from . import register_op


@register_op("DictMapper")
class DictMapper:
    """String/int to index lookup.

    约定：mapping 值从 1 起始，0 保留为 default_idx 表示未命中/缺失。
    下游 Embedding 的 index 0 可固定映射为零向量，避免 padding 与真实特征混淆。
    """

    def __init__(self, mapping: dict[str, int], default_idx: int = 0) -> None:
        """Initialize with mapping table and fallback index."""
        self.mapping = mapping
        self.default_idx = default_idx

    @classmethod
    def from_config(cls, params: dict) -> "DictMapper":
        mapping = {str(k): int(v) for k, v in params.get("mapping", {}).items()}
        return cls(mapping, int(params.get("default_idx", 0)))

    def process(self, inputs: list[Any]) -> Union[int, list[int]]:
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

    def process_batch(self, inputs: list[Any]) -> Union[list[int], list[list[int]]]:
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
