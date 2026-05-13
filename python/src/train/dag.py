"""Feature DAG executor — mirrors src/feats/dag.rs."""
from collections import deque
from dataclasses import dataclass
from typing import Any
from .config import DType, EmbedConfig, FlowConfig, OperatorDef, SourceDef
from .ops import Bucketing, CrossFeature, CustomOp, DictMapper, ExpressionOp, SequenceOp, StringParser

FeatureValue = Any

@dataclass
class FeatureResult:
    features: dict[str, FeatureValue]
    source_names: set[str]
    computed_names: set[str]

class FeatureDag:
    def __init__(self, config: FlowConfig, debug_mode: bool = False):
        self.sources: dict[str, SourceDef] = {}
        for s in config.sources:
            if s.name in self.sources: raise ValueError(f"Duplicate source: '{s.name}'")
            self.sources[s.name] = s
        valid_inputs: set[str] = set(self.sources.keys())
        output_to_provider: dict[str, str] = {}
        for op in config.operators:
            for out in op.outputs:
                if out in valid_inputs: raise ValueError(f"Duplicate output '{out}'")
                valid_inputs.add(out)
                output_to_provider[out] = op.name
        for op in config.operators:
            for inp in op.inputs:
                if inp not in valid_inputs: raise ValueError(f"Unknown input '{inp}' for '{op.name}'")
        self.nodes: dict[str, CustomOp] = {}
        self.node_defs: dict[str, OperatorDef] = {}
        for op_def in config.operators:
            if op_def.name in self.nodes: raise ValueError(f"Duplicate operator: '{op_def.name}'")
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
                if in_degree[neighbor] == 0: queue.append(neighbor)
        if len(execution_order) != len(config.operators):
            raise ValueError("Cycle detected in feature DAG")
        self.execution_order = execution_order
        self.debug_mode = debug_mode

    @staticmethod
    def _parse_default(val_str: str, dtype: DType) -> FeatureValue:
        if dtype.tag == "int": return int(float(val_str))
        elif dtype.tag == "float": return float(val_str)
        elif dtype.tag == "string": return val_str
        elif dtype.tag == "list":
            inner, length = dtype.inner, dtype.length
            if inner.tag == "int": return [int(float(val_str))] * length
            elif inner.tag == "float": return [float(val_str)] * length
            elif inner.tag == "string": return [val_str] * length
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
            return StringParser(str(p.get("sep1", "#")), str(p.get("sep2", "|")), int(p.get("key_index", 0)), int(p.get("pad_len", 0)), str(p.get("pad_val", "unknown")))
        elif op_type == "CrossFeature":
            return CrossFeature(str(p.get("cross_type", "cartesian")))
        elif op_type == "ExpressionOp":
            script = p.get("script")
            if not script: raise ValueError("Missing 'script' for ExpressionOp")
            return ExpressionOp(str(script))
        elif op_type == "SequenceOp":
            return SequenceOp(int(p.get("max_len", 10)), int(p.get("pad_val", 0)))
        else:
            raise ValueError(f"Unsupported operator: {op_type}")

    def embeddable_features(self) -> list[tuple[str, EmbedConfig]]:
        result: list[tuple[str, EmbedConfig]] = []
        for name, src in self.sources.items():
            if src.embed is not None: result.append((name, src.embed))
        for _, op_def in self.node_defs.items():
            if op_def.embed is not None:
                for out_name in op_def.outputs: result.append((out_name, op_def.embed))
        return result

    def execute(self, raw_inputs: dict[str, FeatureValue]) -> FeatureResult:
        context: dict[str, FeatureValue] = {}
        for name, src in self.sources.items():
            context[name] = self._parse_default(src.default_val, src.dtype)
        for name, val in raw_inputs.items(): context[name] = val
        source_names = set(self.sources.keys())
        computed_names: set[str] = set()
        for node_name in self.execution_order:
            op, def_ = self.nodes[node_name], self.node_defs[node_name]
            op_inputs = [context[inp] for inp in def_.inputs]
            output = op.process(op_inputs)
            for out_name in def_.outputs:
                context[out_name] = output
                computed_names.add(out_name)
        if self.debug_mode: self._dump_snapshot(context, source_names, computed_names)
        return FeatureResult(features=context, source_names=source_names, computed_names=computed_names)

    def _dump_snapshot(self, context, source_names, computed_names):
        print("[Feature Snapshot]")
        for name, val in sorted(context.items()):
            origin = "computed" if name in computed_names else "source" if name in source_names else "raw"
            print(f" -> [{origin}] {name:<20} | value={val}")
