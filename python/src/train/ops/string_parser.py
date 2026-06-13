from __future__ import annotations

"""字符串解析算子：结构化字符串的分词与填充。"""
from typing import Any

from . import register_op


@register_op("StringParser")
class StringParser:
    """Parse structured strings into token sequences."""

    def __init__(self, sep1: str, sep2: str, key_index: int, pad_len: int, pad_val: str) -> None:
        """Initialize with delimiters, key index, and padding config."""
        self.sep1 = sep1
        self.sep2 = sep2
        self.key_index = key_index
        self.pad_len = pad_len
        self.pad_val = pad_val

    @classmethod
    def from_config(cls, params: dict) -> "StringParser":
        return cls(
            sep1=str(params.get("sep1", "#")),
            sep2=str(params.get("sep2", "|")),
            key_index=int(params.get("key_index", 0)),
            pad_len=int(params.get("pad_len", 0)),
            pad_val=str(params.get("pad_val", "unknown")),
        )

    def process(self, inputs: list[Any]) -> list[str]:
        """Parse string: split by sep1/sep2, extract key_index field, pad to pad_len."""
        s = str(inputs[0])
        items = s.split(self.sep1) if s else []
        result = []
        for item in items:
            parts = item.split(self.sep2)
            if self.key_index < len(parts):
                result.append(parts[self.key_index])
        if len(result) < self.pad_len:
            result.extend([self.pad_val] * (self.pad_len - len(result)))
        return result[: self.pad_len]

    def process_batch(self, inputs: list[Any]) -> list[list[str]]:
        """Batch parse: N strings -> N lists of parsed tokens."""
        vals = inputs[0]
        if not vals:
            return []
        sep1, sep2 = self.sep1, self.sep2
        ki, pl, pv = self.key_index, self.pad_len, self.pad_val
        results = []
        for val in vals:
            s = str(val) if val else ""
            items = s.split(sep1) if s else []
            result = []
            for item in items:
                parts = item.split(sep2)
                if ki < len(parts):
                    result.append(parts[ki])
            if len(result) < pl:
                result.extend([pv] * (pl - len(result)))
            results.append(result[:pl])
        return results
