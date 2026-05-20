from __future__ import annotations

"""特征哈希算子：DJB2 多种子哈希，Python/Rust 输出一致。"""
from typing import Any


class FeatureHash:
    """Stateless feature hashing with multiple hash functions to reduce collisions.

    Concatenates all inputs with separator, then hashes with k independent
    seed-prefixed DJB2 functions.  Each seed produces one index in [0, vocab_size).
    32-bit wrapping at every step guarantees identical output with Rust.
    """

    def __init__(self, vocab_size: int, num_hashes: int = 1, separator: str = "|"):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.num_hashes = num_hashes
        self.separator = separator

    # ── single-row ──

    def process(self, inputs: list[Any]) -> int | list[int]:
        key = self._build_key(inputs)
        return self._hash_multi(key)

    # ── batch ──

    def process_batch(self, inputs: list[Any]) -> list:
        """Columnar batch: N cols × M rows → M hashed results."""
        if not inputs or not inputs[0]:
            return []
        n = len(inputs[0])
        sep = self.separator
        v = self.vocab_size
        k = self.num_hashes
        results = []
        for i in range(n):
            key = sep.join(str(col[i]) if col[i] is not None else "" for col in inputs)
            if k == 1:
                results.append(_djb2_seeded(key, 0) % v)
            else:
                results.append([_djb2_seeded(key, s) % v for s in range(k)])
        return results

    # ── internal ──

    def _build_key(self, inputs: list[Any]) -> str:
        return self.separator.join(str(v) if v is not None else "" for v in inputs)

    def _hash_multi(self, key: str) -> int | list[int]:
        v = self.vocab_size
        if self.num_hashes == 1:
            return _djb2_seeded(key, 0) % v
        return [_djb2_seeded(key, s) % v for s in range(self.num_hashes)]


# ── module-level for fastest call ──


def _djb2_seeded(key: str, seed: int) -> int:
    """DJB2 with seed prefix and 32-bit wrapping — matches Rust exactly."""
    # inline the seeded key to avoid allocation of format string
    h: int = 5381
    # seed prefix
    for ch in str(seed):
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    h = ((h << 5) + h + ord("_")) & 0xFFFFFFFF
    # key body
    for ch in key:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h & 0x7FFFFFFF
