"""字典映射算子：字符串/整数到索引的映射。"""
from typing import Any


class DictMapper:
    def __init__(self, mapping: dict[str, int], default_idx: int = 0):
        self.mapping = mapping
        self.default_idx = default_idx

    def process(self, inputs: list[Any]) -> int:
        val = inputs[0]
        if isinstance(val, (int, float)):
            val = str(val)
        return (
            self.mapping.get(val, self.default_idx)
            if isinstance(val, str)
            else self.default_idx
        )
