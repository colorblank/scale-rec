from __future__ import annotations

"""特征信息视图：从 DAG 构建结果投影，提供模型构建和广播策略所需的元数据查询。"""

from enum import Enum
from typing import Any

from .config import EmbedConfig, OperatorDef, PoolingMode, SourceDef, SourceKind


class FeatureScope(str, Enum):
    USER = "user"
    ITEM = "item"
    CONTEXT = "context"
    USER_ITEM = "user_item"
    USER_CONTEXT = "user_context"
    ITEM_CONTEXT = "item_context"
    USER_ITEM_CONTEXT = "user_item_context"

    @classmethod
    def from_source_kind(cls, source: SourceKind | None) -> FeatureScope:
        if source is SourceKind.USER:
            return cls.USER
        if source is SourceKind.CONTEXT:
            return cls.CONTEXT
        if source is SourceKind.LABEL:
            raise ValueError("label sources do not have a feature scope")
        return cls.ITEM

    @classmethod
    def combine(cls, scopes: list[FeatureScope]) -> FeatureScope:
        has_user = any(scope.has_user for scope in scopes)
        has_item = any(scope.has_item for scope in scopes)
        has_context = any(scope.has_context for scope in scopes)
        if has_user and has_item and has_context:
            return cls.USER_ITEM_CONTEXT
        if has_user and has_item:
            return cls.USER_ITEM
        if has_user and has_context:
            return cls.USER_CONTEXT
        if has_item and has_context:
            return cls.ITEM_CONTEXT
        if has_user:
            return cls.USER
        if has_context:
            return cls.CONTEXT
        return cls.ITEM

    @property
    def has_user(self) -> bool:
        return self in {self.USER, self.USER_ITEM, self.USER_CONTEXT, self.USER_ITEM_CONTEXT}

    @property
    def has_item(self) -> bool:
        return self in {self.ITEM, self.USER_ITEM, self.ITEM_CONTEXT, self.USER_ITEM_CONTEXT}

    @property
    def has_context(self) -> bool:
        return self in {self.CONTEXT, self.USER_CONTEXT, self.ITEM_CONTEXT, self.USER_ITEM_CONTEXT}


class FeatureInfo:
    def __init__(
        self,
        sources: dict[str, SourceDef],
        node_defs: dict[str, OperatorDef],
        feature_schemas: dict[str, Any],
        execution_order: list[str],
    ) -> None:
        self._sources = sources
        self._node_defs = node_defs
        self._feature_schemas = feature_schemas
        self._execution_order = execution_order

    def embeddable_features(self) -> list[tuple[str, EmbedConfig]]:
        result: list[tuple[str, EmbedConfig]] = []
        for op_def in self._node_defs.values():
            if op_def.embed is not None:
                for out_name in op_def.outputs:
                    schema = self._feature_schemas.get(out_name)
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

    def embed_infos(self) -> dict[str, EmbedConfig]:
        return dict(self.embeddable_features())

    def embed_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.embeddable_features())

    def feature_tuples(self) -> list[tuple[str, int, int]]:
        return [(name, emb.vocab_size, emb.embed_dim) for name, emb in self.embeddable_features()]

    def feature_total_dim(self) -> int:
        total = 0
        for _, emb in self.embeddable_features():
            if emb.pooling is PoolingMode.FLATTEN and emb.seq_len:
                total += emb.embed_dim * emb.seq_len
            else:
                total += emb.embed_dim
        return total

    def feature_pooling(self) -> dict[str, PoolingMode]:
        return {name: emb.pooling for name, emb in self.embeddable_features()}

    def feature_seq_lens(self) -> dict[str, int]:
        return {
            name: emb.seq_len for name, emb in self.embeddable_features() if emb.seq_len is not None
        }

    def op_source_kind(self) -> dict[str, FeatureScope]:
        feat_kind: dict[str, FeatureScope] = {}
        for name, src in self._sources.items():
            feat_kind[name] = FeatureScope.from_source_kind(src.source)
        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            kinds = [feat_kind[inp] for inp in def_.inputs if inp in feat_kind]
            k = FeatureScope.combine(kinds)
            for out_name in def_.outputs:
                feat_kind[out_name] = k
        op_kind: dict[str, FeatureScope] = {}
        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            kinds = [feat_kind[inp] for inp in def_.inputs if inp in feat_kind]
            op_kind[node_name] = FeatureScope.combine(kinds)
        return op_kind

    @property
    def sources(self) -> dict[str, SourceDef]:
        return self._sources

    @property
    def node_defs(self) -> dict[str, OperatorDef]:
        return self._node_defs

    def source_defs(self) -> dict[str, SourceDef]:
        return self._sources

    def source_names(self) -> list[str]:
        return list(self._sources)

    def op_outputs(self, op_name: str) -> list[str] | None:
        def_ = self._node_defs.get(op_name)
        return list(def_.outputs) if def_ else None
