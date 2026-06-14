from __future__ import annotations

"""Feature quality summaries for training batches."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..core.executor import DagExecutor
from ..core.feature_info import FeatureInfo


@dataclass(frozen=True)
class SourceQuality:
    total: int = 0
    missing: int = 0
    default_hits: int = 0

    @property
    def missing_rate(self) -> float:
        return self.missing / self.total if self.total else 0.0

    @property
    def default_rate(self) -> float:
        return self.default_hits / self.total if self.total else 0.0


@dataclass(frozen=True)
class EmbeddableQuality:
    total: int = 0
    empty_sequences: int = 0
    mean_length: float = 0.0
    total_items: int = 0
    padded_items: int = 0
    unique_buckets: int = 0
    bucket_utilization: float = 0.0
    top_buckets: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    @property
    def empty_sequence_rate(self) -> float:
        return self.empty_sequences / self.total if self.total else 0.0

    @property
    def padding_rate(self) -> float:
        return self.padded_items / self.total_items if self.total_items else 0.0


@dataclass(frozen=True)
class FeatureQualityReport:
    rows: int
    sources: dict[str, SourceQuality]
    embeddables: dict[str, EmbeddableQuality]

    def to_metrics(self, prefix: str = "feature_quality") -> dict[str, float]:
        metrics: dict[str, float] = {f"{prefix}.rows": float(self.rows)}
        for name, stat in self.sources.items():
            metrics[f"{prefix}.source.{name}.missing_rate"] = stat.missing_rate
            metrics[f"{prefix}.source.{name}.default_rate"] = stat.default_rate
        for name, stat in self.embeddables.items():
            metrics[f"{prefix}.emb.{name}.empty_sequence_rate"] = stat.empty_sequence_rate
            metrics[f"{prefix}.emb.{name}.mean_length"] = stat.mean_length
            metrics[f"{prefix}.emb.{name}.padding_rate"] = stat.padding_rate
            metrics[f"{prefix}.emb.{name}.bucket_utilization"] = stat.bucket_utilization
        return metrics


def summarize_feature_quality(
    executor: DagExecutor,
    feat_info: FeatureInfo,
    batches: list[dict[str, Any]],
    *,
    max_rows: int = 2000,
    top_k: int = 5,
) -> FeatureQualityReport:
    rows = _collect_rows(batches, max_rows)
    source_stats = _source_quality(feat_info, rows)
    emb_stats = _embeddable_quality(executor, feat_info, rows, top_k)
    return FeatureQualityReport(
        rows=len(rows),
        sources=source_stats,
        embeddables=emb_stats,
    )


def _collect_rows(batches: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        features = batch.get("features", [])
        if isinstance(features, dict):
            n = len(next(iter(features.values()))) if features else 0
            batch_rows = [
                {k: v[i] for k, v in features.items() if v[i] is not None} for i in range(n)
            ]
            rows.extend(batch_rows)
        else:
            rows.extend(features)
        if len(rows) >= max_rows:
            return rows[:max_rows]
    return rows


def _source_quality(feat_info: FeatureInfo, rows: list[dict[str, Any]]) -> dict[str, SourceQuality]:
    from ..core.builder import parse_default

    sources = feat_info.sources
    counters = {name: {"total": 0, "missing": 0, "default_hits": 0} for name in sources}
    defaults = {
        name: parse_default(source.default_val, source.dtype)
        for name, source in sources.items()
    }
    for row in rows:
        for name in sources:
            counters[name]["total"] += 1
            value = row.get(name)
            if _is_missing(value):
                counters[name]["missing"] += 1
                counters[name]["default_hits"] += 1
            elif value == defaults[name]:
                counters[name]["default_hits"] += 1
    return {
        name: SourceQuality(
            total=values["total"],
            missing=values["missing"],
            default_hits=values["default_hits"],
        )
        for name, values in counters.items()
    }


def _embeddable_quality(
    executor: DagExecutor,
    feat_info: FeatureInfo,
    rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, EmbeddableQuality]:
    embed_infos = dict(feat_info.embeddable_features())
    counters: dict[str, Counter[int]] = {name: Counter() for name in embed_infos}
    total = {name: 0 for name in embed_infos}
    empty = {name: 0 for name in embed_infos}
    length_sum = {name: 0 for name in embed_infos}
    total_items = {name: 0 for name in embed_infos}
    padded_items = {name: 0 for name in embed_infos}
    pad_buckets = _embedding_pad_buckets(executor, feat_info)

    for row in rows:
        result = executor.execute(row)
        for name, embed in embed_infos.items():
            value = result.get(name)
            total[name] += 1
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list):
                seq_len = embed.seq_len or len(value)
                tensor_pad = max(seq_len - len(value), 0)
                pad_items = tensor_pad + sum(
                    1 for item in value if _is_padding_bucket(name, item, pad_buckets)
                )
                effective_len = max(len(value) - (pad_items - tensor_pad), 0)
                length_sum[name] += effective_len
                total_items[name] += max(seq_len, len(value))
                padded_items[name] += pad_items
                if effective_len == 0:
                    empty[name] += 1
            else:
                length_sum[name] += 1
                total_items[name] += 1
            for item in values:
                if isinstance(item, int):
                    counters[name][item] += 1

    report = {}
    for name, embed in embed_infos.items():
        unique = len(counters[name])
        report[name] = EmbeddableQuality(
            total=total[name],
            empty_sequences=empty[name],
            mean_length=length_sum[name] / total[name] if total[name] else 0.0,
            total_items=total_items[name],
            padded_items=padded_items[name],
            unique_buckets=unique,
            bucket_utilization=unique / embed.vocab_size if embed.vocab_size else 0.0,
            top_buckets=tuple(counters[name].most_common(top_k)),
        )
    return report


def _embedding_pad_buckets(executor: DagExecutor, feat_info: FeatureInfo) -> dict[str, set[int]]:
    providers = {}
    node_defs = feat_info.node_defs
    for op_name, op_def in node_defs.items():
        for output in op_def.outputs:
            providers[output] = op_name

    nodes = executor.nodes
    result: dict[str, set[int]] = {}
    for feature_name, _ in feat_info.embeddable_features():
        op_name = providers.get(feature_name)
        if op_name is None:
            continue
        op_def = node_defs[op_name]
        op = nodes[op_name]
        pad_values: set[int] = set()
        if op_def.op_type == "SequenceOp":
            pad_values.add(int(op_def.params.get("pad_val", 0)))
        elif op_def.op_type == "ParsedFeatureHash" and hasattr(op, "_hash"):
            pad_values.add(op._hash._hash_one(str(op_def.params.get("pad_val", ""))))
        elif op_def.op_type == "FeatureHash" and hasattr(op, "_hash_one"):
            for input_name in op_def.inputs:
                upstream_name = providers.get(input_name)
                if upstream_name is None:
                    continue
                upstream_def = node_defs[upstream_name]
                if upstream_def.op_type in {
                    "StringParser",
                    "JsonExtractList",
                    "Split",
                    "FlatSplit",
                }:
                    pad_values.add(op._hash_one(str(upstream_def.params.get("pad_val", ""))))
        if pad_values:
            result[feature_name] = pad_values
    return result


def _is_padding_bucket(
    feature_name: str,
    item: Any,
    pad_buckets: dict[str, set[int]],
) -> bool:
    return isinstance(item, int) and item in pad_buckets.get(feature_name, set())


def _is_missing(value: Any) -> bool:
    return value is None or value == ""
