from __future__ import annotations

"""序列算子：填充/截断整数序列至固定长度。"""
from typing import Any


class SequenceOp:
    def __init__(self, max_len: int, pad_val: int = 0):
        self.max_len = max_len
        self.pad_val = pad_val

    def process(self, inputs: list[Any]) -> list[int]:
        seq = [int(x) for x in inputs[0]]
        if len(seq) < self.max_len:
            seq = seq + [self.pad_val] * (self.max_len - len(seq))
        return seq[: self.max_len]
