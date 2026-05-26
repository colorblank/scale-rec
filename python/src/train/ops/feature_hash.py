from __future__ import annotations

"""特征哈希算子：DJB2 多种子哈希，支持逐元素 list 哈希。"""
from typing import Any


class FeatureHash:
    """Stateless feature hashing. List inputs → per-element hash → IntList."""

    def __init__(
        self,
        vocab_size: int,
        num_hashes: int = 1,
        separator: str = "|",
        namespace: str = "",
        salt: str = "",
        version: str = "",
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.num_hashes = num_hashes
        self.separator = separator
        scope_parts = [str(part) for part in (namespace, salt, version) if str(part)]
        self.hash_prefix = "::".join(scope_parts) + "::" if scope_parts else ""

    # ── single-row ──

    def process(self, inputs: list[Any]) -> int | list[int]:
        for v in inputs:
            if isinstance(v, list):
                return [self._hash_one(str(x)) for x in v]
        key = self.separator.join(str(v) if v is not None else "" for v in inputs)
        return self._hash_multi(key)

    # ── batch ──

    def process_batch(self, inputs: list[Any]) -> list[int] | list[list[int]]:
        """Columnar batch: list 列逐元素 hash → IntList 列。"""
        if not inputs or not inputs[0]:
            return []
        n = len(inputs[0])

        # 检测第一列是否有 list 值
        first_col = inputs[0]
        is_list_col = any(isinstance(first_col[i], list) for i in range(min(n, 3)))

        results = []
        for i in range(n):
            if is_list_col:
                # 逐元素 hash → list[int] 输出
                row_vals = []
                for col in inputs:
                    val = col[i] if i < len(col) else None
                    if isinstance(val, list):
                        row_vals.extend(str(x) if x is not None else "" for x in val)
                    elif val is not None:
                        row_vals.append(str(val))
                results.append([self._hash_one(x) for x in row_vals])
            else:
                key = self.separator.join(
                    str(col[i]) if col[i] is not None else "" for col in inputs
                )
                results.append(self._hash_multi(key))
        return results

    # ── internal ──

    def _hash_one(self, key: str, seed: int = 0) -> int:
        return _djb2_seeded(f"{self.hash_prefix}{key}", seed) % self.vocab_size

    def _hash_multi(self, key: str) -> int | list[int]:
        if self.num_hashes == 1:
            return self._hash_one(key, 0)
        return [self._hash_one(key, s) for s in range(self.num_hashes)]


def _djb2_seeded(key: str, seed: int) -> int:
    """DJB2 with seed prefix and 32-bit wrapping — matches Rust exactly."""
    h: int = 5381
    for ch in str(seed):
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    h = ((h << 5) + h + ord("_")) & 0xFFFFFFFF
    for b in key.encode("utf-8"):
        h = ((h << 5) + h + b) & 0xFFFFFFFF
    return h & 0x7FFFFFFF
