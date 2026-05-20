from __future__ import annotations

"""字符串拼接算子：多输入 → 单字符串输出。"""

from typing import Any


class StringConcat:
    """将多路输入拼接为字符串。输入取值类型不限，输出为 str。

    Args:
        separator: 拼接分隔符，默认 "_"。
    """

    def __init__(self, separator: str = "_") -> None:
        self.separator = separator

    def process(self, inputs: list[Any]) -> str:
        return self.separator.join(str(v) if v is not None else "" for v in inputs)

    def process_batch(self, inputs: list[Any]) -> list[str]:
        """Columnar batch: N cols × M rows → M concatenated strings."""
        if not inputs or not inputs[0]:
            return []
        n = len(inputs[0])
        sep = self.separator
        results = []
        for i in range(n):
            key = sep.join(str(col[i]) if col[i] is not None else "" for col in inputs)
            results.append(key)
        return results
