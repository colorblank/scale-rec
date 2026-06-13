from __future__ import annotations

"""融合预处理算子：多输入拼接后直接 hash。"""

from typing import Any

from . import register_op
from .feature_hash import FeatureHash


@register_op("ConcatHash")
class ConcatHash:
    def __init__(
        self,
        vocab_size: int,
        *,
        separator: str = "_",
        num_hashes: int = 1,
        namespace: str = "",
        salt: str = "",
        version: str = "",
    ) -> None:
        self._hash = FeatureHash(
            vocab_size,
            num_hashes=num_hashes,
            separator=separator,
            namespace=namespace,
            salt=salt,
            version=version,
        )

    @classmethod
    def from_config(cls, params: dict) -> "ConcatHash":
        return cls(
            vocab_size=int(params.get("vocab_size", 1000)),
            num_hashes=int(params.get("num_hashes", 1)),
            separator=str(params.get("separator", "_")),
            namespace=str(params.get("namespace", "")),
            salt=str(params.get("salt", "")),
            version=str(params.get("version", "")),
        )

    def process(self, inputs: list[Any]) -> int | list[int]:
        return self._hash.process(inputs)

    def process_batch(self, inputs: list[Any]) -> list[int] | list[list[int]]:
        return self._hash.process_batch(inputs)
