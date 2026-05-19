from __future__ import annotations

"""JSON 数组提取算子：解析 JSON 数组/对象数组并提取内容。"""
import json
from typing import Any, Optional


class JsonExtractList:
    """Parse JSON array strings and extract elements or specific properties."""

    def __init__(self, key: Optional[str], pad_len: int, pad_val: str):
        self.key = key
        self.pad_len = pad_len
        self.pad_val = pad_val

    def process(self, inputs: list[Any]) -> list[str]:
        val = inputs[0]
        s = str(val) if val else ""
        result = []
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

        if len(result) < self.pad_len:
            result.extend([self.pad_val] * (self.pad_len - len(result)))
        return result[: self.pad_len]

    def process_batch(self, inputs: list[Any]) -> list[list[str]]:
        vals = inputs[0]
        if not vals:
            return []
        key, pl, pv = self.key, self.pad_len, self.pad_val
        results = []
        for val in vals:
            s = str(val) if val else ""
            result = []
            if s:
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        for item in arr:
                            if key is not None:
                                if isinstance(item, dict) and key in item:
                                    result.append(str(item[key]))
                            else:
                                result.append(str(item))
                except json.JSONDecodeError:
                    pass

            if len(result) < pl:
                result.extend([pv] * (pl - len(result)))
            results.append(result[:pl])
        return results
