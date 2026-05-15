from __future__ import annotations

"""字符串拼接哈希算子：两个字符串拼接 → hash 映射到固定词表。"""
import os
from typing import Any

import yaml


def _djb2(s: str) -> int:
    h = 5381
    for ch in s:
        h = ((h << 5) + h) + ord(ch)
    return h & 0x7FFFFFFF


class StringConcatHash:
    """Concat two strings with separator, hash to fixed vocab range.

    Training mode: assign new keys to [0, vocab_size - oov_reserve),
    record mapping to hash_map_path.
    Inference mode: load mapping from hash_map_path, OOV keys
    hash to reserved range [vocab_size - oov_reserve, vocab_size).
    """

    def __init__(
        self,
        vocab_size: int,
        oov_reserve: int = 0,
        hash_map_path: str = "",
        mode: str = "train",
        separator: str = "|",
    ):
        self.vocab_size = vocab_size
        self.oov_reserve = oov_reserve
        self.hash_map_path = hash_map_path
        self.mode = mode
        self.separator = separator
        self._mapping: dict[str, int] = {}
        self._next_idx = 0
        self._main_size = vocab_size - oov_reserve

        if mode == "inference" and hash_map_path and os.path.exists(hash_map_path):
            with open(hash_map_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            if raw and "mapping" in raw:
                self._mapping = raw["mapping"]
            self._next_idx = len(self._mapping)

    def _save_mapping(self) -> None:
        if self.hash_map_path:
            os.makedirs(os.path.dirname(self.hash_map_path) or ".", exist_ok=True)
            with open(self.hash_map_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"mapping": self._mapping}, f)

    def save_mapping(self) -> None:
        """Explicitly persist hash mapping. Called after training, not per-key."""
        self._save_mapping()

    def process(self, inputs: list[Any]) -> int:
        s1 = str(inputs[0]) if inputs[0] is not None else ""
        s2 = str(inputs[1]) if inputs[1] is not None else ""
        key = f"{s1}{self.separator}{s2}"

        if self.mode == "inference":
            if key in self._mapping:
                return self._mapping[key]
            # OOV: deterministic hash to reserved range
            return (_djb2(key) % self.oov_reserve) + self._main_size

        # Training mode
        if key not in self._mapping:
            if self._next_idx >= self._main_size:
                return (_djb2(key) % self.oov_reserve) + self._main_size
            self._mapping[key] = self._next_idx
            self._next_idx += 1
        return self._mapping[key]
