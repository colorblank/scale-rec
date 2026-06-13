from __future__ import annotations

"""字符串列表切分提取算子：对 StrList 中每个字符串进行 split 并提取指定索引内容。"""
from typing import Any

from . import register_op


@register_op("ListStringParser")
class ListStringParser:
    """Parse list of strings, splitting each by a separator and extracting a specific index."""

    def __init__(self, sep: str, key_index: int) -> None:
        self.sep = sep
        self.key_index = key_index

    @classmethod
    def from_config(cls, params: dict) -> "ListStringParser":
        return cls(
            sep=str(params.get("sep", ",")),
            key_index=int(params.get("key_index", 0)),
        )

    def process(self, inputs: list[Any]) -> list[str]:
        str_list = inputs[0]
        if not isinstance(str_list, list):
            raise ValueError("ListStringParser requires a list of strings as input")

        result = []
        for item in str_list:
            parts = str(item).split(self.sep)
            if self.key_index < len(parts):
                result.append(parts[self.key_index])
            else:
                result.append("")
        return result

    def process_batch(self, inputs: list[Any]) -> list[list[str]]:
        list_of_lists = inputs[0]
        if not list_of_lists:
            return []

        sep, key_index = self.sep, self.key_index
        results = []
        for str_list in list_of_lists:
            if not isinstance(str_list, list):
                raise ValueError("ListStringParser requires a list of strings as input")
            result = []
            for item in str_list:
                parts = str(item).split(sep)
                if key_index < len(parts):
                    result.append(parts[key_index])
                else:
                    result.append("")
            results.append(result)
        return results
