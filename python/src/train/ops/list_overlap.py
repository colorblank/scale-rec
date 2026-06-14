from __future__ import annotations

"""列表重叠检测算子：判断两个列表是否存在共同元素。"""
from typing import Any

from . import register_op


@register_op("ListOverlap")
class ListOverlap:
    """Check if two lists have overlapping elements. Returns 1 if any overlap, 0 otherwise."""

    @classmethod
    def from_config(cls, params: dict) -> ListOverlap:
        return cls()

    def process(self, inputs: list[Any]) -> int:
        a, b = inputs[0], inputs[1]
        set_a = {str(x) for x in a if x and str(x).strip()} if isinstance(a, list) else set()
        set_b = {str(x) for x in b if x and str(x).strip()} if isinstance(b, list) else set()
        if not set_a or not set_b:
            return 0
        return 1 if set_a & set_b else 0

    def process_batch(self, inputs: list[Any]) -> list[int]:
        """Batch overlap: N pairs of lists -> N flags."""
        list_a, list_b = inputs[0], inputs[1]
        result = []
        for a, b in zip(list_a, list_b, strict=False):
            sa = {str(x) for x in a if x and str(x).strip()} if isinstance(a, list) else set()
            sb = {str(x) for x in b if x and str(x).strip()} if isinstance(b, list) else set()
            if not sa or not sb:
                result.append(0)
            else:
                result.append(1 if sa & sb else 0)
        return result
