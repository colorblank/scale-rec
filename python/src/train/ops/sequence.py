from __future__ import annotations

"""序列算子：填充/截断整数序列至固定长度。"""
from typing import Any


class SequenceOp:
    """Pad or truncate integer sequences to fixed length."""

    def __init__(self, max_len: int, pad_val: int = 0) -> None:
        """Initialize with target length and padding value."""
        self.max_len = max_len
        self.pad_val = pad_val

    def process(self, inputs: list[Any]) -> list[int]:
        """Pad or truncate integer sequence to max_len."""
        seq = [int(x) for x in inputs[0]]
        if len(seq) < self.max_len:
            seq = seq + [self.pad_val] * (self.max_len - len(seq))
        return seq[: self.max_len]

    def process_batch(self, inputs: list[Any]) -> list[list[int]]:
        """Batch padding: N sequences → N padded sequences."""
        vals = inputs[0]
        if not vals:
            return []
        max_len, pad_val = self.max_len, self.pad_val
        result = []
        for val in vals:
            seq = [int(x) for x in val]
            if len(seq) < max_len:
                seq = seq + [pad_val] * (max_len - len(seq))
            result.append(seq[:max_len])
        return result
