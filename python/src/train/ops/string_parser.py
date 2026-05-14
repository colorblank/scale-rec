from __future__ import annotations

"""字符串解析算子：结构化字符串的分词与填充。"""
from typing import Any


class StringParser:
    """Parse structured strings into token sequences."""

    def __init__(self, sep1: str, sep2: str, key_index: int, pad_len: int, pad_val: str):
        """Initialize with delimiters, key index, and padding config."""
        self.sep1 = sep1
        self.sep2 = sep2
        self.key_index = key_index
        self.pad_len = pad_len
        self.pad_val = pad_val

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
