from __future__ import annotations

"""Small bounded LRU cache for hot training-side preprocessing operators."""

from collections import OrderedDict
from collections.abc import Hashable
from typing import Generic, TypeVar

V = TypeVar("V")


class LruCache(Generic[V]):
    def __init__(self, max_size: int = 100_000) -> None:
        self.max_size = max(max_size, 0)
        self._values: OrderedDict[Hashable, V] = OrderedDict()

    def get(self, key: Hashable) -> tuple[bool, V | None]:
        if self.max_size <= 0:
            return False, None
        if key not in self._values:
            return False, None
        value = self._values.pop(key)
        self._values[key] = value
        return True, value

    def put(self, key: Hashable, value: V) -> None:
        if self.max_size <= 0:
            return
        if key in self._values:
            self._values.pop(key)
        self._values[key] = value
        while len(self._values) > self.max_size:
            self._values.popitem(last=False)

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)
