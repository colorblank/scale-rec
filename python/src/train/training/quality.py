from __future__ import annotations

"""Feature quality summaries for training batches."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from ..core.config import OpType
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
    truncations: int = 0
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
    def truncation_rate(self) -> float:
        return self.truncations / self.total if self.total else 0.0

    @property
    def padding_rate(self) -> float:
        return self.padded_items / self.total_items if self.total_items else 0.0


@dataclass(frozen=True)
class FeatureHashCacheStats:
    total: int = 0
    hits: int = 0
    misses: int = 0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.total if self.total else 0.0


@dataclass(frozen=True)
class FeatureQualityReport:
    rows: int
    sources: dict[str, SourceQuality]
    embeddables: dict[str, EmbeddableQuality]
    hash_cache: dict[str, FeatureHashCacheStats] = field(default_factory=dict)

    def to_metrics(self, prefix: str = "feature_quality") -> dict[str, float]:
        metrics: dict[str, float] = {f"{prefix}.rows": float(self.rows)}
        for name, stat in self.sources.items():
            metrics[f"{prefix}.source.{name}.missing_rate"] = stat.missing_rate
            metrics[f"{prefix}.source.{name}.default_rate"] = stat.default_rate
        for name, stat in self.embeddables.items():
            metrics[f"{prefix}.emb.{name}.empty_sequence_rate"] = stat.empty_sequence_rate
            metrics[f"{prefix}.emb.{name}.truncation_rate"] = stat.truncation_rate
            metrics[f"{prefix}.emb.{name}.mean_length"] = stat.mean_length
            metrics[f"{prefix}.emb.{name}.padding_rate"] = stat.padding_rate
            metrics[f"{prefix}.emb.{name}.bucket_utilization"] = stat.bucket_utilization
        for name, stat in self.hash_cache.items():
            metrics[f"{prefix}.hash_cache.{name}.total"] = float(stat.total)
            metrics[f"{prefix}.hash_cache.{name}.hit_rate"] = stat.hit_rate
            metrics[f"{prefix}.hash_cache.{name}.miss_rate"] = stat.miss_rate
            metrics[f"{prefix}.hash_cache.{name}.cache_size"] = float(stat.cache_size)
        return metrics


class EmbeddingBucketTracker:
    """Accumulate exact embedding lookups from supervised training batches."""

    def __init__(self, feat_info: FeatureInfo) -> None:
        self._vocab_sizes = {
            name: embed.vocab_size for name, embed in feat_info.embeddable_features()
        }
        providers = _provider_map(feat_info)
        self._feature_metadata = {}
        for name in self._vocab_sizes:
            op_def = feat_info.node_defs[providers[name]]
            metadata: dict[str, Any] = {"operator_type": op_def.op_type.value}
            if op_def.op_type is OpType.DICT_MAPPER:
                metadata["default_idx"] = int(op_def.params.get("default_idx", 0))
            self._feature_metadata[name] = metadata
        self._counts = {
            name: torch.zeros(vocab_size, dtype=torch.int64)
            for name, vocab_size in self._vocab_sizes.items()
        }
        self.steps = 0

    def update(self, features: dict[str, torch.Tensor]) -> None:
        for name, counts in self._counts.items():
            values = features[name].detach().reshape(-1).cpu()
            counts.add_(torch.bincount(values, minlength=counts.numel())[: counts.numel()])
        self.steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "counts": {name: counts.clone() for name, counts in self._counts.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_counts = state.get("counts", {})
        for name, counts in self._counts.items():
            saved = saved_counts.get(name)
            if saved is not None:
                if tuple(saved.shape) != tuple(counts.shape):
                    raise ValueError(
                        f"embedding bucket tracker shape mismatch for '{name}': "
                        f"{tuple(saved.shape)} != {tuple(counts.shape)}"
                    )
                counts.copy_(saved)
        self.steps = int(state.get("steps", 0))

    def report(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for name, counts in self._counts.items():
            active = int(torch.count_nonzero(counts).item())
            vocab_size = counts.numel()
            features[name] = {
                **self._feature_metadata[name],
                "vocab_size": vocab_size,
                "total_hits": int(counts.sum().item()),
                "active_buckets": active,
                "inactive_buckets": vocab_size - active,
                "bucket_utilization": active / vocab_size if vocab_size else 0.0,
                "inactive_bucket_ids": torch.nonzero(counts == 0).reshape(-1).tolist(),
                "bucket_hits": counts.tolist(),
            }
        return {
            "schema_version": 1,
            "training_steps": self.steps,
            "features": features,
        }


def write_embedding_bucket_report(report: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, sort_keys=False, allow_unicode=True)
    return path


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
    emb_stats, hash_cache_stats = _embeddable_quality(executor, feat_info, rows, top_k)
    return FeatureQualityReport(
        rows=len(rows),
        sources=source_stats,
        embeddables=emb_stats,
        hash_cache=hash_cache_stats,
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
        name: parse_default(source.default_val, source.dtype) for name, source in sources.items()
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
) -> tuple[dict[str, EmbeddableQuality], dict[str, FeatureHashCacheStats]]:
    embed_infos = dict(feat_info.embeddable_features())
    counters: dict[str, Counter[int]] = {name: Counter() for name in embed_infos}
    total = dict.fromkeys(embed_infos, 0)
    empty = dict.fromkeys(embed_infos, 0)
    truncations = dict.fromkeys(embed_infos, 0)
    length_sum = dict.fromkeys(embed_infos, 0)
    total_items = dict.fromkeys(embed_infos, 0)
    padded_items = dict.fromkeys(embed_infos, 0)
    pad_buckets = _embedding_pad_buckets(executor, feat_info)

    truncation_limits = _embedding_truncation_limits(executor, feat_info, embed_infos)
    hash_ops = _enable_hash_cache_stats(executor, feat_info, embed_infos)

    for row in rows:
        result = executor.execute(row)
        for name, embed in embed_infos.items():
            value = result.get(name)
            total[name] += 1
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list):
                seq_len = embed.seq_len or len(value)
                limit = truncation_limits.get(name)
                if limit is not None and len(value) >= limit:
                    truncations[name] += 1
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

    hash_cache = _read_hash_cache_stats(hash_ops)

    report = {}
    for name, embed in embed_infos.items():
        unique = len(counters[name])
        report[name] = EmbeddableQuality(
            total=total[name],
            empty_sequences=empty[name],
            truncations=truncations[name],
            mean_length=length_sum[name] / total[name] if total[name] else 0.0,
            total_items=total_items[name],
            padded_items=padded_items[name],
            unique_buckets=unique,
            bucket_utilization=unique / embed.vocab_size if embed.vocab_size else 0.0,
            top_buckets=tuple(counters[name].most_common(top_k)),
        )
    return report, hash_cache


def _provider_map(feat_info: FeatureInfo) -> dict[str, str]:
    providers: dict[str, str] = {}
    for op_name, op_def in feat_info.node_defs.items():
        for output in op_def.outputs:
            providers[output] = op_name
    return providers


def _embedding_truncation_limits(
    executor: DagExecutor,
    feat_info: FeatureInfo,
    embed_infos: dict[str, Any],
) -> dict[str, int]:
    providers = _provider_map(feat_info)
    node_defs = feat_info.node_defs
    limits: dict[str, int] = {}
    for name, embed in embed_infos.items():
        if embed.seq_len is not None:
            limits[name] = embed.seq_len
        else:
            op_name = providers.get(name)
            if op_name is not None:
                op_def = node_defs.get(op_name)
                if op_def is not None:
                    limit = op_def.params.get("max_len") or op_def.params.get("pad_len")
                    if limit is not None and int(limit) > 0:
                        limits[name] = int(limit)
    return limits


def _enable_hash_cache_stats(
    executor: DagExecutor,
    feat_info: FeatureInfo,
    embed_infos: dict[str, Any],
) -> dict[str, Any]:
    providers = _provider_map(feat_info)
    node_defs = feat_info.node_defs
    nodes = executor.nodes
    ops: dict[str, Any] = {}
    for name in embed_infos:
        op_name = providers.get(name)
        if op_name is None:
            continue
        op_def = node_defs.get(op_name)
        if op_def is None:
            continue
        if op_def.op_type is OpType.FEATURE_HASH:
            op = nodes.get(op_name)
            if op is not None and hasattr(op, "enable_cache_stats"):
                op.enable_cache_stats()
                ops[name] = op
        elif op_def.op_type is OpType.PARSED_FEATURE_HASH:
            op = nodes.get(op_name)
            inner = getattr(op, "_hash", None)
            if inner is not None and hasattr(inner, "enable_cache_stats"):
                inner.enable_cache_stats()
                ops[name] = inner
    return ops


def _read_hash_cache_stats(ops: dict[str, Any]) -> dict[str, FeatureHashCacheStats]:
    result: dict[str, FeatureHashCacheStats] = {}
    for name, op in ops.items():
        stats = op.disable_cache_stats() if hasattr(op, "disable_cache_stats") else None
        if stats is not None:
            result[name] = FeatureHashCacheStats(
                total=stats["total"],
                hits=stats["hits"],
                misses=stats["misses"],
                cache_size=stats["cache_size"],
            )
    return result


def _embedding_pad_buckets(executor: DagExecutor, feat_info: FeatureInfo) -> dict[str, set[int]]:
    providers = _provider_map(feat_info)
    node_defs = feat_info.node_defs
    nodes = executor.nodes
    result: dict[str, set[int]] = {}
    for feature_name, _ in feat_info.embeddable_features():
        op_name = providers.get(feature_name)
        if op_name is None:
            continue
        op_def = node_defs[op_name]
        op = nodes[op_name]
        pad_values: set[int] = set()
        if op_def.op_type is OpType.SEQUENCE_OP:
            pad_values.add(int(op_def.params.get("pad_val", 0)))
        elif op_def.op_type is OpType.PARSED_FEATURE_HASH and hasattr(op, "_hash"):
            pad_values.add(op._hash._hash_one(str(op_def.params.get("pad_val", ""))))
        elif op_def.op_type is OpType.FEATURE_HASH and hasattr(op, "_hash_one"):
            for input_name in op_def.inputs:
                upstream_name = providers.get(input_name)
                if upstream_name is None:
                    continue
                upstream_def = node_defs[upstream_name]
                if upstream_def.op_type in {
                    OpType.STRING_PARSER,
                    OpType.JSON_EXTRACT_LIST,
                    OpType.SPLIT,
                    OpType.FLAT_SPLIT,
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
