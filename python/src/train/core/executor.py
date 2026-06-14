from __future__ import annotations

"""DAG 执行器：ExecutionPlan + DagExecutor，统一 plan-based 执行路径。"""

from dataclasses import dataclass, field
from typing import Any

from ..ops import CustomOp

Fv = Any


@dataclass
class ExecStep:
    op_idx: int
    input_cols: list[int]
    output_cols: list[int]


@dataclass
class ExecutionPlan:
    steps: list[ExecStep] = field(default_factory=list)
    source_cols: list[int] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)
    col_names: list[str | None] = field(default_factory=list)
    source_defaults: list[Fv] = field(default_factory=list)
    col_count: int = 0
    embed_ids: list[int] = field(default_factory=list)
    _ops: list[CustomOp] = field(default_factory=list)

    def execute_plan(
        self,
        columns: dict[str, list],
        skip_op_idx: set[int] | None = None,
        precomputed: dict[int, Fv] | None = None,
    ) -> list[list]:
        skip_op_idx = skip_op_idx or set()
        precomputed = precomputed or {}
        n_rows = len(next(iter(columns.values()))) if columns else 0
        if n_rows == 0:
            return []

        context: list[list] = [[] for _ in range(self.col_count)]

        for i in range(len(self.source_cols)):
            cid = self.source_cols[i]
            name = self.source_names[i]
            default = self.source_defaults[i] if i < len(self.source_defaults) else 0
            col = columns.get(name)
            if col is not None and len(col) == n_rows:
                context[cid] = list(col)
            else:
                context[cid] = [default] * n_rows

        for col_id, val in precomputed.items():
            if col_id < len(context):
                context[col_id] = [val] * n_rows

        for step in self.steps:
            if step.op_idx in skip_op_idx:
                continue
            op = self._ops[step.op_idx]
            input_slices = [context[cid] for cid in step.input_cols]
            result = op.process_batch(input_slices) if hasattr(op, "process_batch") else None

            if result is None:
                result = []
                for i in range(n_rows):
                    row_inputs = [col[i] for col in input_slices]
                    result.append(op.process(row_inputs))

            for cid in step.output_cols:
                context[cid] = result

        return context


class DagExecutor:
    def __init__(
        self,
        plan: ExecutionPlan,
        sources: dict[str, Any],
        node_defs: dict[str, Any] | None = None,
        execution_order: list[str] | None = None,
    ) -> None:
        self._plan = plan
        self._sources = sources
        self._node_defs = node_defs or {}
        self._execution_order = execution_order or []

    def execute_plan(
        self,
        columns: dict[str, list],
        skip_op_idx: set[int] | None = None,
        precomputed: dict[int, Fv] | None = None,
    ) -> list[list]:
        return self._plan.execute_plan(columns, skip_op_idx, precomputed)

    def execute_batch(self, columns: dict[str, list]) -> dict[str, list]:
        n_rows = len(next(iter(columns.values()))) if columns else 0
        if n_rows == 0:
            return {}

        context: dict[str, list] = dict(columns)
        for name, src in self._sources.items():
            if name not in context:
                from .builder import parse_default

                default = parse_default(src.default_val, src.dtype)
                context[name] = [default] * n_rows
            else:
                col = context[name]
                if any(v is None for v in col):
                    from .builder import parse_default

                    default = parse_default(src.default_val, src.dtype)
                    context[name] = [default if v is None else v for v in col]

        name_to_op: dict[str, Any] = {}
        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            for step in self._plan.steps:
                step_outputs = [self._plan.col_names[cid] for cid in step.output_cols]
                if def_.outputs and any(o == step_outputs for o in [def_.outputs]):
                    name_to_op[node_name] = self._plan._ops[step.op_idx]
                    break
            if node_name not in name_to_op:
                name_to_op[node_name] = None

        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            op = name_to_op.get(node_name)
            if op is None:
                continue
            op_inputs = [context[inp] for inp in def_.inputs]
            if hasattr(op, "process_batch"):
                output = op.process_batch(op_inputs)
            else:
                output = []
                for i in range(n_rows):
                    row_inputs = [col[i] for col in op_inputs]
                    output.append(op.process(row_inputs))
            if def_.outputs:
                for out_name in def_.outputs:
                    context[out_name] = output

        return context

    @property
    def nodes(self) -> dict[str, CustomOp]:
        """映射 op_name → CustomOp 实例。"""
        if not hasattr(self, "__nodes_cache"):
            self.__nodes_cache = {}
            for i, node_name in enumerate(self._execution_order):
                self.__nodes_cache[node_name] = self._plan._ops[i]
        return self.__nodes_cache

    def execute(self, raw_inputs: dict[str, Any]) -> dict[str, Any]:
        """单行执行 DAG，返回 context 字典。"""
        context: dict[str, Any] = {
            name: val for name, val in raw_inputs.items() if name in self._sources
        }
        for name, src in self._sources.items():
            if name not in context:
                from .builder import parse_default

                context[name] = parse_default(src.default_val, src.dtype)
        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            op = self.nodes[node_name]
            op_inputs = [context[inp] for inp in def_.inputs]
            output = op.process(op_inputs)
            for out_name in def_.outputs:
                context[out_name] = output
        return context

    def plan(self) -> ExecutionPlan:
        return self._plan

    def source_defs(self) -> dict[str, Any]:
        return self._sources
