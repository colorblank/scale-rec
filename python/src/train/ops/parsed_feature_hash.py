from __future__ import annotations

"""融合预处理算子：先解析为 token 序列，再逐 token hash。"""

import json
from typing import Any, Optional

from . import register_op
from .feature_hash import FeatureHash


@register_op("ParsedFeatureHash")
class ParsedFeatureHash:
    """Parse structured inputs and hash tokens in one operator.

    Supported modes:
    - ``json``: JSON array / array of objects -> token list
    - ``structured``: string parser with ``sep1`` / ``sep2``
    - ``split``: plain string split
    - ``list_split``: split each item in a list and keep the same list length
    - ``flat_split``: flatten a list of strings after splitting each item
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        parse_mode: str,
        num_hashes: int = 1,
        separator: str = "|",
        namespace: str = "",
        salt: str = "",
        version: str = "",
        key: Optional[str] = None,
        sep1: str = "|",
        sep2: str = "#",
        key_index: int = 0,
        sep: str = ",",
        max_len: int = 0,
        pad_len: int = 0,
        pad_val: str = "",
    ) -> None:
        if num_hashes != 1:
            raise ValueError("ParsedFeatureHash only supports num_hashes=1 for list outputs")
        self.parse_mode = parse_mode
        self.key = key
        self.sep1 = sep1
        self.sep2 = sep2
        self.key_index = key_index
        self.sep = sep
        self.max_len = max_len
        self.pad_len = pad_len
        self.pad_val = pad_val
        self._hash = FeatureHash(
            vocab_size,
            num_hashes=num_hashes,
            separator=separator,
            namespace=namespace,
            salt=salt,
            version=version,
        )

    @classmethod
    def from_config(cls, params: dict) -> "ParsedFeatureHash":
        return cls(
            vocab_size=int(params.get("vocab_size", 1000)),
            parse_mode=str(params.get("parse_mode", "json")),
            num_hashes=int(params.get("num_hashes", 1)),
            separator=str(params.get("separator", "|")),
            namespace=str(params.get("namespace", "")),
            salt=str(params.get("salt", "")),
            version=str(params.get("version", "")),
            key=params.get("key"),
            sep1=str(params.get("sep1", "|")),
            sep2=str(params.get("sep2", "#")),
            key_index=int(params.get("key_index", 0)),
            sep=str(params.get("sep", ",")),
            max_len=int(params.get("max_len", 0)),
            pad_len=int(params.get("pad_len", 0)),
            pad_val=str(params.get("pad_val", "")),
        )

    def process(self, inputs: list[Any]) -> list[int]:
        return self._hash.process([self._parse(inputs[0])])

    def process_batch(self, inputs: list[Any]) -> list[list[int]]:
        vals = inputs[0]
        if not vals:
            return []
        return [self._hash.process([self._parse(val)]) for val in vals]

    def _parse(self, value: Any) -> list[str]:
        if self.parse_mode == "json":
            return self._parse_json(value)
        if self.parse_mode == "structured":
            return self._parse_structured(value)
        if self.parse_mode == "structured_flat_split":
            return self._parse_structured_flat_split(value)
        if self.parse_mode == "split":
            return self._normalize_max((str(value) if value is not None else "").split(self.sep))
        if self.parse_mode == "list_split":
            return self._parse_list_split(value)
        if self.parse_mode == "flat_split":
            return self._parse_flat_split(value)
        raise ValueError(f"Unsupported ParsedFeatureHash mode: {self.parse_mode}")

    def _parse_json(self, value: Any) -> list[str]:
        s = str(value) if value else ""
        result: list[str] = []
        if s:
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    for item in arr:
                        if self.key is not None:
                            if isinstance(item, dict) and self.key in item:
                                result.append(str(item[self.key]))
                        else:
                            result.append(str(item))
            except json.JSONDecodeError:
                pass
        return self._normalize(result)

    def _parse_structured(self, value: Any) -> list[str]:
        s = str(value) if value else ""
        result: list[str] = []
        if s:
            for item in s.split(self.sep1):
                parts = item.split(self.sep2)
                if self.key_index < len(parts):
                    result.append(parts[self.key_index])
        return self._normalize(result)

    def _parse_list_split(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return self._normalize([])
        result: list[str] = []
        for item in value:
            parts = str(item).split(self.sep) if item is not None else []
            if self.key_index < len(parts):
                result.append(parts[self.key_index])
            else:
                result.append("")
        return self._normalize(result)

    def _parse_flat_split(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return self._normalize_max([])
        result: list[str] = []
        for item in value:
            if item:
                result.extend(str(item).split(self.sep))
        return self._normalize_max(result)

    def _parse_structured_flat_split(self, value: Any) -> list[str]:
        s = str(value) if value else ""
        result: list[str] = []
        if s:
            for item in s.split(self.sep1):
                parts = item.split(self.sep2)
                if self.key_index < len(parts):
                    token = parts[self.key_index]
                    if token:
                        result.extend(token.split(self.sep))
        return self._normalize_max(result)

    def _normalize(self, parts: list[str]) -> list[str]:
        if self.pad_len <= 0:
            return parts
        parts = parts[: self.pad_len]
        if len(parts) < self.pad_len:
            parts.extend([self.pad_val] * (self.pad_len - len(parts)))
        return parts

    def _normalize_max(self, parts: list[str]) -> list[str]:
        if self.max_len <= 0:
            return parts
        parts = parts[: self.max_len]
        if len(parts) < self.max_len:
            parts.extend([self.pad_val] * (self.max_len - len(parts)))
        return parts
