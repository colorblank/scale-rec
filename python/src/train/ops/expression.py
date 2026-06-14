from __future__ import annotations

"""表达式算子：使用 AST 安全解析执行脚本表达式。"""
import ast
import math
import operator
from typing import Any

from . import register_op

_FUNCTIONS = {"log": math.log, "abs": abs, "max": max, "min": min, "sqrt": math.sqrt}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


@register_op("ExpressionOp")
class ExpressionOp:
    """Evaluate script expressions using AST parsing (no eval)."""

    def __init__(self, script: str) -> None:
        self.tree = ast.parse(script.strip(), mode="eval")

    @classmethod
    def from_config(cls, params: dict) -> ExpressionOp:
        script = params.get("script")
        if not script:
            raise ValueError("Missing 'script' for ExpressionOp")
        return cls(str(script))

    def process(self, inputs: list[Any]) -> float:
        scope = {f"v{i}": float(val) for i, val in enumerate(inputs)}
        scope.update(_FUNCTIONS)
        return float(self._eval(self.tree.body, scope))

    def _eval(self, node: ast.AST, scope: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return scope[node.id]
        if isinstance(node, ast.UnaryOp):
            return _UNARYOPS[type(node.op)](self._eval(node.operand, scope))
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](
                self._eval(node.left, scope), self._eval(node.right, scope)
            )
        if isinstance(node, ast.Call):
            func = self._eval(node.func, scope)
            args = [self._eval(a, scope) for a in node.args]
            return func(*args)
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")
