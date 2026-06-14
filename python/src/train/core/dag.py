from __future__ import annotations

"""特征 DAG 执行器 —  facade，委派给 builder / executor / feature_info / preprocessor。

新代码请直接使用 builder、executor、feature_info、preprocessor 模块，而非此 facade。
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..ops import CustomOp, create_op
from .builder import (
    DagBuilder,
    ValidationReport,
    parse_default,
)

_build = DagBuilder.build
from .config import EmbedConfig, FlowConfig, OperatorDef, SourceDef
from .executor import DagExecutor
from .feature_info import FeatureInfo
from .preprocessor import DagPreprocessor
from .schema import FeatureSchema

logger = logging.getLogger(__name__)

FeatureValue = Any


@dataclass
class FeatureResult:
    features: dict[str, FeatureValue]
    source_names: set[str]
    computed_names: set[str]


class FeatureDag:
    def __init__(
        self,
        config: FlowConfig,
        debug_mode: bool = False,
        tracer: Any | None = None,
        strict_validation: bool = False,
    ) -> None:
        artifact = _build(config)
        self._artifact = artifact
        self._executor = DagExecutor(
            artifact.plan,
            artifact.sources,
            node_defs=artifact.node_defs,
            execution_order=artifact.execution_order,
        )
        self._feat_info = FeatureInfo(
            artifact.sources,
            artifact.node_defs,
            artifact.feature_schemas,
            artifact.execution_order,
        )
        self._preprocessor = DagPreprocessor(self._feat_info)

        self.sources: dict[str, SourceDef] = artifact.sources
        self.nodes: dict[str, CustomOp] = {}
        op_name_to_idx: dict[str, int] = {}
        for i, node_name in enumerate(artifact.execution_order):
            op = artifact.plan._ops[i]
            self.nodes[node_name] = op
            op_name_to_idx[node_name] = i
        self.node_defs: dict[str, OperatorDef] = artifact.node_defs
        self.execution_order: list[str] = artifact.execution_order
        self.debug_mode = debug_mode
        self.tracer = tracer
        self.feature_schemas: dict[str, FeatureSchema] = artifact.feature_schemas
        self.validation_report: ValidationReport = artifact.validation_report
        self._source_names = tuple(artifact.sources)
        self._source_name_set = set(artifact.sources)
        self._embed_infos = dict(self._feat_info.embeddable_features())
        self._embed_names = tuple(self._embed_infos)

        if strict_validation and self.validation_report.warnings:
            details = ", ".join(issue.message for issue in self.validation_report.warnings)
            raise ValueError(f"strict validation failed: {details}")

    def _validate(self, config: FlowConfig) -> ValidationReport:
        return self.validation_report

    @property
    def executor(self) -> DagExecutor:
        return self._executor

    @property
    def feat_info(self) -> FeatureInfo:
        return self._feat_info

    @property
    def preprocessor(self) -> DagPreprocessor:
        return self._preprocessor

    @staticmethod
    def _parse_default(val_str: str, dtype: Any) -> FeatureValue:
        return parse_default(val_str, dtype)

    @staticmethod
    def _create_op(def_: OperatorDef) -> CustomOp:
        return create_op(def_.op_type, def_.params)

    def embeddable_features(self) -> list[tuple[str, EmbedConfig]]:
        return self._feat_info.embeddable_features()

    def feature_tuples(self) -> list[tuple[str, int, int]]:
        return self._feat_info.feature_tuples()

    def feature_total_dim(self) -> int:
        return self._feat_info.feature_total_dim()

    def feature_pooling(self) -> dict[str, str]:
        return self._feat_info.feature_pooling()

    def feature_seq_lens(self) -> dict[str, int]:
        return self._feat_info.feature_seq_lens()

    def execute_batch(self, columns: dict[str, list]) -> dict[str, list]:
        n_rows = len(next(iter(columns.values()))) if columns else 0
        if n_rows == 0:
            return {}

        context: dict[str, list] = {}
        for name, col in columns.items():
            if name in self.sources:
                context[name] = list(col)

        for name, src in self.sources.items():
            if name not in context:
                default = self._parse_default(src.default_val, src.dtype)
                context[name] = [default] * n_rows
            else:
                col = context[name]
                if any(v is None for v in col):
                    default = self._parse_default(src.default_val, src.dtype)
                    context[name] = [default if v is None else v for v in col]

        for node_name in self.execution_order:
            op, def_ = self.nodes[node_name], self.node_defs[node_name]
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

    def preprocess_batch(self, rows: list[dict] | dict[str, list]) -> dict[str, torch.Tensor]:
        if isinstance(rows, dict):
            columns = {
                name: list(values) for name, values in rows.items() if name in self._source_name_set
            }
        else:
            n_rows = len(rows)
            if n_rows > 0:
                import pandas as pd
                df = pd.DataFrame(rows)
                columns = {
                    col: df[col].tolist() for col in df.columns if col in self._source_name_set
                }
            else:
                columns = {}

        if self.tracer:
            n_rows = len(next(iter(columns.values()))) if columns else 0
            for i in range(n_rows):
                self.tracer.begin_sample(i)

        result = self.execute_batch(columns)

        if self.tracer:
            n_rows = len(next(iter(columns.values()))) if columns else 0
            for _ in range(n_rows):
                self.tracer.end_sample()

        return self._preprocessor.preprocess(result)

    def execute(self, raw_inputs: dict[str, FeatureValue], sample_id: int = 0) -> FeatureResult:
        context: dict[str, FeatureValue] = {
            name: val for name, val in raw_inputs.items() if name in self.sources
        }

        overridden = list(raw_inputs.keys())
        for name, src in self.sources.items():
            if name not in context:
                context[name] = self._parse_default(src.default_val, src.dtype)

        if self.tracer:
            self.tracer.trace_defaults(context)
            self.tracer.trace_overrides(context, overridden)

        source_names = set(self.sources.keys())
        computed_names: set[str] = set()

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
