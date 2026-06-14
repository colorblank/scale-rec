from __future__ import annotations

"""字符串分割算子：单字符串 → 按分隔符切分为定长列表。"""

from typing import Any

from . import register_op


@register_op("Split")
class Split:
    """将输入字符串按分隔符切分，截断/填充到定长。

    Args:
        sep: 分隔符，默认 "|"。
        max_len: 最大长度，0 表示不限制。超出截断，不足填充 pad_val。
        pad_val: 填充值，默认 ""。
    """

    def __init__(self, sep: str = "|", max_len: int = 0, pad_val: str = "") -> None:
        self.sep = sep
        self.max_len = max_len
        self.pad_val = pad_val

    @classmethod
    def from_config(cls, params: dict) -> Split:
        return cls(
            sep=str(params.get("sep", "|")),
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
        s = str(inputs[0]) if inputs[0] is not None else ""
        parts = s.split(self.sep) if s else []
        return self._normalize(parts)

    def process_batch(self, inputs: list[Any]) -> list[list[str]]:
        """Columnar batch: 1 col × M rows → M lists."""
        if not inputs or not inputs[0]:
            return []
        n = len(inputs[0])
        sep = self.sep
        results = []
        for i in range(n):
            s = str(inputs[0][i]) if inputs[0][i] is not None else ""
            parts = s.split(sep) if s else []
            results.append(self._normalize(parts))
        return results
