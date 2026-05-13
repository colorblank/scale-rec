import math
from typing import Any

_SAFE = {"log": math.log, "abs": abs, "max": max, "min": min, "sqrt": math.sqrt}


class ExpressionOp:
    def __init__(self, script: str):
        self.script = script

    def process(self, inputs: list[Any]) -> float:
        scope = {}
        for i, val in enumerate(inputs):
            scope[f"v{i}"] = float(val)
        scope.update(_SAFE)
        return float(eval(self.script, {"__builtins__": {}}, scope))
