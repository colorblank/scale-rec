from __future__ import annotations

"""DAG 构建器：FlowConfig → 拓扑排序 → 校验 → 预编译 ExecutionPlan。"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..ops import CustomOp, create_op

Fv = Any
from .config import DType, FlowConfig, OpType, Role, SourceDef, parse_float_strict, parse_int_strict
from .executor import ExecStep, ExecutionPlan
from .schema import FeatureSchema, infer_feature_schemas

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    feature: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    source_count: int
    embeddable_count: int
    intermediate_count: int

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")


@dataclass
class DagArtifact:
    sources: dict[str, SourceDef] = field(default_factory=dict)
    node_defs: dict[str, Any] = field(default_factory=dict)
    execution_order: list[str] = field(default_factory=list)
    plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    feature_schemas: dict[str, FeatureSchema] = field(default_factory=dict)
    data_sources: list[Any] = field(default_factory=list)
    validation_report: ValidationReport = field(
        default_factory=lambda: ValidationReport(
            issues=(),
            source_count=0,
            embeddable_count=0,
            intermediate_count=0,
        )
    )


def parse_default(val_str: str, dtype: DType) -> Fv:
    if dtype.tag == "int":
        return parse_int_strict(val_str)
    elif dtype.tag == "float":
        return parse_float_strict(val_str)
    elif dtype.tag == "string":
        return val_str
    elif dtype.tag == "enum":
        candidate = dtype.default if dtype.default is not None else val_str
        if dtype.values and candidate in dtype.values:
            return candidate
        if dtype.oov is not None:
            return dtype.oov
        return candidate
    elif dtype.tag == "list":
        inner, length = dtype.inner, dtype.max_len or dtype.length
        if inner is None:
            raise ValueError("list dtype requires inner dtype")
        if inner.tag == "int":
            return [parse_int_strict(val_str)] * length
        elif inner.tag == "float":
            return [parse_float_strict(val_str)] * length
        elif inner.tag == "string":
            return [val_str] * length
        elif inner.tag == "enum":
            candidate = inner.default if inner.default is not None else val_str
            if inner.values and candidate not in inner.values and inner.oov is not None:
                candidate = inner.oov
            return [candidate] * length
    return 0


def _create_op(op_type: OpType, params: Any) -> CustomOp:
    return create_op(op_type, params)


def _validate(
    sources: dict[str, SourceDef],
    operators: list[Any],
    embeddable: set[str],
) -> ValidationReport:
    downstream_consumers: set[str] = set()
    for op in operators:
        downstream_consumers.update(op.inputs)

    orphan_sources = []
    orphan_outputs = []
    intermediate = 0
    issues = []

    for s in operators:
        if hasattr(s, "role") and s.role != Role.FEATURE:
            continue
    for name in sources:
        if name not in downstream_consumers and name not in embeddable:
            orphan_sources.append(name)
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="orphan_source",
                    message=f"source '{name}' is not consumed and not embeddable",
                    feature=name,
                )
            )

    for op in operators:
        for out_name in op.outputs:
            if out_name in embeddable:
                continue
            if out_name in downstream_consumers:
                intermediate += 1
                continue
            orphan_outputs.append(f"{op.name} -> {out_name}")
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="orphan_output",
                    message=f"operator '{op.name}' output '{out_name}' has no consumer and no embed",
                    feature=out_name,
                )
            )

    total_features = len(sources)
    logger.info(
        "sources: consumed=%d/%d orphan=%d",
        total_features - len(orphan_sources),
        total_features,
        len(orphan_sources),
    )
    logger.info(
        "outputs: embeddable=%d intermediate=%d orphan=%d",
        len(embeddable),
        intermediate,
        len(orphan_outputs),
    )
    if orphan_sources:
        logger.warning("orphan sources: %s", orphan_sources)
    if orphan_outputs:
        logger.warning("orphan outputs: %s", orphan_outputs)
    return ValidationReport(
        issues=tuple(issues),
        source_count=total_features,
        embeddable_count=len(embeddable),
        intermediate_count=intermediate,
    )


class DagBuilder:
    @staticmethod
    def build(config: FlowConfig) -> DagArtifact:
        feature_schemas = infer_feature_schemas(config)
        data_sources = list(config.data_sources)

        sources: dict[str, SourceDef] = {}
        for s in config.sources:
            if s.role != Role.FEATURE:
                logger.info("source '%s' skipped (role=%s)", s.name, s.role)
                continue
            if s.name in sources:
                raise ValueError(f"Duplicate source name: '{s.name}'")
            sources[s.name] = s

        valid_inputs: set[str] = set(sources.keys())
        output_to_provider: dict[str, str] = {}
        for op in config.operators:
            for out in op.outputs:
                if out in valid_inputs:
                    raise ValueError(f"Duplicate output '{out}'")
                valid_inputs.add(out)
                output_to_provider[out] = op.name
        for op in config.operators:
            for inp in op.inputs:
                if inp not in valid_inputs:
                    raise ValueError(f"Unknown input '{inp}' for '{op.name}'")

        nodes: dict[str, CustomOp] = {}
        node_defs: dict[str, Any] = {}
        for op_def in config.operators:
            if op_def.name in nodes:
                raise ValueError(f"Duplicate operator: '{op_def.name}'")
            nodes[op_def.name] = _create_op(op_def.op_type, op_def.params)
            node_defs[op_def.name] = op_def

        in_degree = {op.name: 0 for op in config.operators}
        adjacency: dict[str, list[str]] = {op.name: [] for op in config.operators}
        for op in config.operators:
            for inp in op.inputs:
                provider = output_to_provider.get(inp)
                if provider:
                    adjacency[provider].append(op.name)
                    in_degree[op.name] += 1
        queue = deque([n for n, d in in_degree.items() if d == 0])
        execution_order: list[str] = []
        while queue:
            node = queue.popleft()
            execution_order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(execution_order) != len(config.operators):
            raise ValueError("Cycle detected in feature DAG")

        col_id: dict[str, int] = {}
        source_cols: list[int] = []
        source_names: list[str] = []
        source_defaults: list[Fv] = []
        col_names: list[str | None] = [None] * len(sources)
        for i, source_name in enumerate(sources):
            col_id[source_name] = i
            source_cols.append(i)
            source_names.append(source_name)
            col_names[i] = source_name
            source_def = sources[source_name]
            source_defaults.append(parse_default(source_def.default_val, source_def.dtype))
        col_count = len(sources)
        for op_def in config.operators:
            for out in op_def.outputs:
                if out not in col_id:
                    col_id[out] = col_count
                    col_names.append(out)
                    col_count += 1

        op_name_to_idx: dict[str, int] = {}
        plan_ops: list[CustomOp] = []
        for i, node_name in enumerate(execution_order):
            op_name_to_idx[node_name] = i
            plan_ops.append(nodes.pop(node_name))

        steps: list[ExecStep] = []
        for node_name in execution_order:
            def_ = node_defs[node_name]
            op_idx = op_name_to_idx[node_name]
            input_cols = [col_id[inp] for inp in def_.inputs]
            output_cols = [col_id[out] for out in def_.outputs]
            steps.append(ExecStep(op_idx=op_idx, input_cols=input_cols, output_cols=output_cols))

        embed_pairs: list[tuple[str, int]] = []
        for op_def in config.operators:
            if op_def.embed is not None:
                embed_pairs.extend(
                    (out_name, col_id[out_name])
                    for out_name in op_def.outputs
                    if out_name in col_id
                )
        embed_pairs.sort(key=lambda x: x[0])
        embed_ids = [cid for _, cid in embed_pairs]

        embeddable = {name for name, _ in embed_pairs}
        validation_report = _validate(sources, config.operators, embeddable)

        plan = ExecutionPlan(
            steps=steps,
            source_cols=source_cols,
            source_names=source_names,
            col_names=col_names,
            source_defaults=source_defaults,
            col_count=col_count,
            embed_ids=embed_ids,
            _ops=plan_ops,
        )

        return DagArtifact(
            sources=sources,
            node_defs=node_defs,
            execution_order=execution_order,
            plan=plan,
            feature_schemas=feature_schemas,
            data_sources=data_sources,
            validation_report=validation_report,
        )
