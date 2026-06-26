from __future__ import annotations

"""特征哈希算子：DJB2 多种子哈希，支持逐元素 list 哈希。"""

from typing import Any

from . import register_op
from .cache import LruCache

DEFAULT_HASH_CACHE_SIZE = 100_000


@register_op("FeatureHash")
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
        self._cache_stats: dict[str, int] | None = None
        self._hash_cache: LruCache[int] = LruCache(DEFAULT_HASH_CACHE_SIZE)

    @classmethod
    def from_config(cls, params: dict) -> FeatureHash:
        return cls(
            vocab_size=int(params.get("vocab_size", 1000)),
            num_hashes=int(params.get("num_hashes", 1)),
            separator=str(params.get("separator", "|")),
            namespace=str(params.get("namespace", "")),
            salt=str(params.get("salt", "")),
            version=str(params.get("version", "")),
        )

    # ── single-row ──

    def process(self, inputs: list[Any]) -> int | list[int]:
        row_vals: list[str] = []
        has_list = False
        for v in inputs:
            if isinstance(v, list):
                has_list = True
                row_vals.extend(str(x) if x is not None else "" for x in v)
            elif v is not None:
                row_vals.append(str(v))
        if has_list:
            return [self._hash_one(x) for x in row_vals]
        key = self.separator.join(str(v) if v is not None else "" for v in inputs)
        return self._hash_multi(key)

    # ── batch ──

    def process_batch(self, inputs: list[Any]) -> list[int] | list[list[int]]:
        """Columnar batch: list 列逐元素 hash → IntList 列。"""
        if not inputs or not inputs[0]:
            return []
        n = len(inputs[0])
        row_has_list = [
            any(isinstance(col[i], list) for col in inputs if i < len(col)) for i in range(n)
        ]
        has_list_row = any(row_has_list)
        has_scalar_row = any(not flag for flag in row_has_list)
        if has_list_row and has_scalar_row:
            raise ValueError("mixed scalar/list rows are not supported in FeatureHash batch")

        results = []
        for i in range(n):
            if has_list_row:
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

    def enable_cache_stats(self) -> None:
        self._cache_stats = {"total": 0, "hits": 0, "misses": 0}
        self._hash_cache.clear()

    def read_cache_stats(self) -> dict[str, int] | None:
        if self._cache_stats is None:
            return None
        return {
            "total": self._cache_stats["total"],
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "cache_size": len(self._hash_cache),
        }

    def disable_cache_stats(self) -> dict[str, int] | None:
        result = self.read_cache_stats()
        self._cache_stats = None
        return result

    def _hash_one(self, key: str, seed: int = 0) -> int:
        cache_key = (key, seed)
        hit, cached = self._hash_cache.get(cache_key)
        if self._cache_stats is not None:
            self._cache_stats["total"] += 1
            if hit:
                self._cache_stats["hits"] += 1
            else:
                self._cache_stats["misses"] += 1
        if hit and cached is not None:
            return cached
        value = _djb2_seeded(f"{self.hash_prefix}{key}", seed) % self.vocab_size
        self._hash_cache.put(cache_key, value)
        return value

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
