from __future__ import annotations

"""列表打平分割算子：StrList 输入 → 每个元素按分隔符切分后打平为单层列表。"""

from typing import Any

from . import register_op


@register_op("FlatSplit")
class FlatSplit:
    """将字符串列表中每个元素按分隔符切分，打平为定长列表。

    Args:
        sep: 分隔符，默认 ","。
        max_len: 最大长度，0 表示不限制。
        pad_val: 填充值，默认 ""。
    """

    def __init__(self, sep: str = ",", max_len: int = 0, pad_val: str = "") -> None:
        self.sep = sep
        self.max_len = max_len
        self.pad_val = pad_val

    @classmethod
    def from_config(cls, params: dict) -> "FlatSplit":
        return cls(
            sep=str(params.get("sep", ",")),
            max_len=int(params.get("max_len", 0)),
            pad_val=str(params.get("pad_val", "")),
        )

    def _normalize(self, parts: list[str]) -> list[str]:
        if self.max_len <= 0:
            return parts
        parts = parts[: self.max_len]
        while len(parts) < self.max_len:
            parts.append(self.pad_val)
        return parts

    def process(self, inputs: list[Any]) -> list[str]:
        str_list = inputs[0] if inputs[0] is not None else []
        if not isinstance(str_list, list):
            return self._normalize([])
        all_parts: list[str] = []
        for s in str_list:
            if s:
                all_parts.extend(str(s).split(self.sep))
        return self._normalize(all_parts)

    def process_batch(self, inputs: list[Any]) -> list[list[str]]:
        """Columnar batch: 1 col (list[list[str]]) × M rows → M flat lists."""
        if not inputs or not inputs[0]:
            return []
        n = len(inputs[0])
        sep = self.sep
        results = []
        for i in range(n):
            str_list = inputs[0][i] if inputs[0][i] is not None else []
            if not isinstance(str_list, list):
                results.append(self._normalize([]))
                continue
            all_parts: list[str] = []
            for s in str_list:
                if s:
                    all_parts.extend(str(s).split(sep))
            results.append(self._normalize(all_parts))
        return results
