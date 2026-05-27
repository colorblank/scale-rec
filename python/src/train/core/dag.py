from __future__ import annotations

"""特征 DAG 执行器 — mirrors src/feats/dag.rs."""
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

import torch

from .config import (
    DType,
    EmbedConfig,
    FlowConfig,
    OperatorDef,
    Role,
    SourceDef,
    parse_float_strict,
    parse_int_strict,
)
from ..ops import (
    Bucketing,
    CrossFeature,
    CustomOp,
    DictMapper,
    ExpressionOp,
    FeatureHash,
    FlatSplit,
    JsonExtractList,
    ListOverlap,
    ListStringParser,
    SequenceOp,
    Split,
    StringConcat,
    StringParser,
)
from .schema import FeatureSchema, infer_feature_schemas

if False:  # pragma: no cover
    from ..debug.tracer import DebugTracer

FeatureValue = Any


@dataclass
class FeatureResult:
    features: dict[str, FeatureValue]
    source_names: set[str]
    computed_names: set[str]


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


class FeatureDag:
    def __init__(
        self,
        config: FlowConfig,
        debug_mode: bool = False,
        tracer: "DebugTracer | None" = None,
        strict_validation: bool = False,
    ) -> None:
        self.sources: dict[str, SourceDef] = {}
        self.tracer = tracer  # Optional[DebugTracer] for per-stage I/O tracing
        for s in config.sources:
            if s.role != Role.FEATURE:
                logger.info("source '%s' skipped (role=%s)", s.name, s.role)
                continue
            if s.name in self.sources:
                raise ValueError(f"Duplicate source: '{s.name}'")
            self.sources[s.name] = s
        valid_inputs: set[str] = set(self.sources.keys())
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
        self.nodes: dict[str, CustomOp] = {}
        self.node_defs: dict[str, OperatorDef] = {}
        for op_def in config.operators:
            if op_def.name in self.nodes:
                raise ValueError(f"Duplicate operator: '{op_def.name}'")
            self.nodes[op_def.name] = self._create_op(op_def)
            self.node_defs[op_def.name] = op_def
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
        self.execution_order = execution_order
        self.debug_mode = debug_mode
        self.feature_schemas: dict[str, FeatureSchema] = infer_feature_schemas(config)
        self.validation_report = self._validate(config)
        self._source_names = tuple(self.sources)
        self._source_name_set = set(self.sources)
        self._embed_infos = dict(self.embeddable_features())
        self._embed_names = tuple(self._embed_infos)
        if strict_validation and self.validation_report.warnings:
            details = ", ".join(issue.message for issue in self.validation_report.warnings)
            raise ValueError(f"strict validation failed: {details}")

    def _validate(self, config: FlowConfig) -> ValidationReport:
        """校验 DAG 完整性：source 消费率、输出利用率、推理跳过算子。"""
        # 下游消费者集合
        downstream_consumers: set[str] = set()
        for op in config.operators:
            downstream_consumers.update(op.inputs)

        # 算子输出分类
        embeddable = {name for name, _ in self.embeddable_features()}
        orphan_sources = []
        orphan_outputs = []
        intermediate = 0
        issues = []

        for s in config.sources:
            if s.role != Role.FEATURE:
                continue
            if s.name not in downstream_consumers and s.name not in embeddable:
                orphan_sources.append(s.name)
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="orphan_source",
                        message=f"source '{s.name}' is not consumed and not embeddable",
                        feature=s.name,
                    )
                )

        for op in config.operators:
            for out_name in op.outputs:
                if out_name in embeddable:
                    continue  # 送入模型
                if out_name in downstream_consumers:
                    intermediate += 1  # 被下游算子消费
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

        total_features = sum(1 for s in config.sources if s.role == Role.FEATURE)
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

    @staticmethod
    def _parse_default(val_str: str, dtype: DType) -> FeatureValue:
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

    @staticmethod
    def _create_op(def_: OperatorDef) -> CustomOp:
        p, op_type = def_.params, def_.op_type
        if op_type == "DictMapper":
            mapping = {str(k): int(v) for k, v in p.get("mapping", {}).items()}
            return DictMapper(mapping, int(p.get("default_idx", 0)))
        elif op_type == "Bucketing":
            return Bucketing([float(x) for x in p.get("boundaries", [])])
        elif op_type == "StringParser":
            return StringParser(
                str(p.get("sep1", "#")),
                str(p.get("sep2", "|")),
                int(p.get("key_index", 0)),
                int(p.get("pad_len", 0)),
                str(p.get("pad_val", "unknown")),
            )
        elif op_type == "JsonExtractList":
            return JsonExtractList(
                p.get("key"),
                int(p.get("pad_len", 0)),
                str(p.get("pad_val", "")),
            )
        elif op_type == "ListStringParser":
            return ListStringParser(
                str(p.get("sep", ",")),
                int(p.get("key_index", 0)),
            )
        elif op_type == "CrossFeature":
            return CrossFeature(str(p.get("cross_type", "cartesian")))
        elif op_type == "ExpressionOp":
            script = p.get("script")
            if not script:
                raise ValueError("Missing 'script' for ExpressionOp")
            return ExpressionOp(str(script))
        elif op_type == "SequenceOp":
            return SequenceOp(int(p.get("max_len", 10)), int(p.get("pad_val", 0)))
        elif op_type == "Split":
            return Split(
                str(p.get("sep", "|")),
                int(p.get("max_len", 0)),
                str(p.get("pad_val", "")),
            )
        elif op_type == "FlatSplit":
            return FlatSplit(
                str(p.get("sep", ",")),
                int(p.get("max_len", 0)),
                str(p.get("pad_val", "")),
            )
        elif op_type == "ListOverlap":
            return ListOverlap()
        elif op_type == "StringConcat":
            return StringConcat(str(p.get("separator", "_")))
        elif op_type == "FeatureHash":
            return FeatureHash(
                vocab_size=int(p.get("vocab_size", 1000)),
                num_hashes=int(p.get("num_hashes", 1)),
                separator=str(p.get("separator", "|")),
                namespace=str(p.get("namespace", "")),
                salt=str(p.get("salt", "")),
                version=str(p.get("version", "")),
            )
        else:
            raise ValueError(f"Unsupported operator: {op_type}")

    def embeddable_features(self) -> list[tuple[str, EmbedConfig]]:
        result: list[tuple[str, EmbedConfig]] = []
        for _, op_def in self.node_defs.items():
            if op_def.embed is not None:
                for out_name in op_def.outputs:
                    schema = self.feature_schemas.get(out_name)
                    if schema and schema.dtype.is_list and op_def.embed.seq_len is None:
                        emb = EmbedConfig(
                            vocab_size=op_def.embed.vocab_size,
                            embed_dim=op_def.embed.embed_dim,
                            pooling=op_def.embed.pooling,
                            seq_len=schema.dtype.length,
                            truncation=op_def.embed.truncation,
                        )
                    else:
                        emb = op_def.embed
                    result.append((out_name, emb))
        result.sort(key=lambda x: x[0])
        return result

    def feature_tuples(self) -> list[tuple[str, int, int]]:
        """返回特征元组 (name, vocab_size, base_embed_dim)，不做 flatten 膨胀。"""
        return [(name, emb.vocab_size, emb.embed_dim) for name, emb in self.embeddable_features()]

    def feature_total_dim(self) -> int:
        """考虑 pooling 策略后的总输出维度。"""
        total = 0
        for _, emb in self.embeddable_features():
            if emb.pooling == "flatten" and emb.seq_len:
                total += emb.embed_dim * emb.seq_len
            else:
                total += emb.embed_dim
        return total

    def feature_pooling(self) -> dict[str, str]:
        """返回 {feature_name: pooling_strategy} 映射。"""
        return {name: emb.pooling for name, emb in self.embeddable_features()}

    def feature_seq_lens(self) -> dict[str, int]:
        """返回配置了固定序列长度的 embedding 特征。"""
        return {
            name: emb.seq_len for name, emb in self.embeddable_features() if emb.seq_len is not None
        }

    def execute_batch(self, columns: dict[str, list]) -> dict[str, list]:
        """Execute DAG on columnar batch data. Each value in `columns` is a list of N elements.

        Operators with `process_batch()` are called once per column.
        Operators without it fall back to row-by-row `process()`.
        """
        n_rows = len(next(iter(columns.values()))) if columns else 0
        if n_rows == 0:
            return {}

        context: dict[str, list] = {}

        # Stage 1: raw inputs first
        for name, col in columns.items():
            if name in self.sources:
                context[name] = list(col)

        # Stage 2: defaults for missing
        for name, src in self.sources.items():
            if name not in context:
                default = self._parse_default(src.default_val, src.dtype)
                context[name] = [default] * n_rows
            else:
                col = context[name]
                if any(v is None for v in col):
                    default = self._parse_default(src.default_val, src.dtype)
                    context[name] = [default if v is None else v for v in col]

        # Stage 3: execute operators in topological order (bulk)
        for node_name in self.execution_order:
            op, def_ = self.nodes[node_name], self.node_defs[node_name]
            op_inputs = [context[inp] for inp in def_.inputs]

            if hasattr(op, "process_batch"):
                output = op.process_batch(op_inputs)
            else:
                # Fallback: row-by-row
                output = []
                for i in range(n_rows):
                    row_inputs = [col[i] for col in op_inputs]
                    output.append(op.process(row_inputs))

            if def_.outputs:
                for out_name in def_.outputs:
                    context[out_name] = output

        return context

    def preprocess_batch(self, rows: list[dict] | dict[str, list]) -> dict[str, torch.Tensor]:
        """Execute DAG and build feature tensors. List-valued features with
        pooling!=flatten are stacked as 2D (batch, seq_len) tensors.
        """
        if isinstance(rows, dict):
            columns = {
                name: list(values)
                for name, values in rows.items()
                if name in self._source_name_set
            }
            n_rows = len(next(iter(columns.values()))) if columns else 0
        else:
            n_rows = len(rows)
            columns = {name: [None] * n_rows for name in self._source_names}
            seen: set[str] = set()
            for i, row in enumerate(rows):
                for name, val in row.items():
                    if name in self._source_name_set:
                        columns[name][i] = val
                        seen.add(name)
            columns = {name: col for name, col in columns.items() if name in seen}

        if self.tracer:
            for i in range(n_rows):
                self.tracer.begin_sample(i)

        result = self.execute_batch(columns)

        # Build pooling lookup from embed configs
        embed_infos = self._embed_infos
        embed_names = self._embed_names

        feature_lists: dict[str, list] = {name: [] for name in embed_names}
        for i in range(n_rows):
            for name in embed_names:
                col = result.get(name)
                if col is not None and i < len(col):
                    val = col[i]
                else:
                    val = 0
                pooling = embed_infos[name].pooling
                if pooling == "first" and isinstance(val, list):
                    val = val[0] if val else 0
                feature_lists[name].append(val)

        if self.tracer:
            for _ in range(n_rows):
                self.tracer.end_sample()

        tensors: dict[str, torch.Tensor] = {}
        for name, vals in feature_lists.items():
            pooling = embed_infos[name].pooling
            if pooling != "first":
                if not vals:
                    tensors[name] = torch.tensor([], dtype=torch.long)
                    continue
                if not isinstance(vals[0], list):
                    raise ValueError(
                        f"feature '{name}' pooling '{pooling}' requires list-valued inputs"
                    )
                if pooling == "flatten":
                    seq_len = embed_infos[name].seq_len
                    if not seq_len or seq_len <= 0:
                        raise ValueError(f"feature '{name}' pooling flatten requires seq_len > 0")
                else:
                    seq_len = embed_infos[name].seq_len
                    if not seq_len or seq_len <= 0:
                        raise ValueError(
                            f"feature '{name}' pooling '{pooling}' requires fixed max_len > 0"
                        )
                trunc = embed_infos[name].truncation
                if trunc == "tail":
                    padded = [v[-seq_len:] + [0] * max(seq_len - len(v), 0) for v in vals]
                else:
                    padded = [v[:seq_len] + [0] * max(seq_len - len(v), 0) for v in vals]
                tensors[name] = torch.tensor(padded, dtype=torch.long)
            else:
                tensors[name] = torch.tensor(vals, dtype=torch.long)
        return tensors

    def execute(self, raw_inputs: dict[str, FeatureValue], sample_id: int = 0) -> FeatureResult:
        context: dict[str, FeatureValue] = {}

        # Stage 1: raw inputs first (avoid allocating defaults that will be overwritten)
        for name, val in raw_inputs.items():
            if name in self.sources:
                context[name] = val

        # Stage 2: fill defaults only for missing keys
        overridden = list(raw_inputs.keys())
        for name, src in self.sources.items():
            if name not in context:
                context[name] = self._parse_default(src.default_val, src.dtype)

        if self.tracer:
            self.tracer.trace_defaults(context)
            self.tracer.trace_overrides(context, overridden)

        source_names = set(self.sources.keys())
        computed_names: set[str] = set()

        # Stage 3: execute operators in topological order
        for node_name in self.execution_order:
            op, def_ = self.nodes[node_name], self.node_defs[node_name]
            op_inputs = [context[inp] for inp in def_.inputs]
            output = op.process(op_inputs)
            if self.tracer:
                self.tracer.trace_operator(node_name, def_.inputs, op_inputs, def_.outputs, output)
            for out_name in def_.outputs:
                context[out_name] = output
                computed_names.add(out_name)

        if self.debug_mode:
            self._dump_snapshot(context, source_names, computed_names)
        if self.tracer:
            self.tracer.end_sample()
        return FeatureResult(
            features=context,
            source_names=source_names,
            computed_names=computed_names,
        )

    def _dump_snapshot(
        self,
        context: dict[str, FeatureValue],
        source_names: set[str],
        computed_names: set[str],
    ) -> None:
        logger.debug("[Feature Snapshot]")
        for name, val in sorted(context.items()):
            origin = (
                "computed"
                if name in computed_names
                else "source"
                if name in source_names
                else "raw"
            )
            logger.debug(" -> [%s] %-20s | value=%s", origin, name, val)
