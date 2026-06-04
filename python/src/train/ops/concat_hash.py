from __future__ import annotations

"""融合预处理算子：多输入拼接后直接 hash。"""

from typing import Any

from .feature_hash import FeatureHash


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

    def process(self, inputs: list[Any]) -> int | list[int]:
        return self._hash.process(inputs)

    def process_batch(self, inputs: list[Any]) -> list[int] | list[list[int]]:
        return self._hash.process_batch(inputs)
