from __future__ import annotations

"""表达式算子：受限 eval 执行脚本表达式。"""
import math
from typing import Any

_SAFE = {"log": math.log, "abs": abs, "max": max, "min": min, "sqrt": math.sqrt}


class ExpressionOp:
    """Evaluate script expressions with restricted eval."""

    def __init__(self, script: str):
        """Initialize with expression script string."""
        self.script = script

    def process(self, inputs: list[Any]) -> float:
        """Evaluate script with inputs bound as v0..vN. Supports log, abs, max, min, sqrt."""
        scope = {}
        for i, val in enumerate(inputs):
            scope[f"v{i}"] = float(val)
        scope.update(_SAFE)
        return float(eval(self.script, {"__builtins__": {}}, scope))
