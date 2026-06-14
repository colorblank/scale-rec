from __future__ import annotations

"""特征信息视图：从 DAG 构建结果投影，提供模型构建和广播策略所需的元数据查询。"""

from typing import Any

from .config import EmbedConfig, OperatorDef, SourceDef


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
        for _, op_def in self._node_defs.items():
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
            if emb.pooling == "flatten" and emb.seq_len:
                total += emb.embed_dim * emb.seq_len
            else:
                total += emb.embed_dim
        return total

    def feature_pooling(self) -> dict[str, str]:
        return {name: emb.pooling for name, emb in self.embeddable_features()}

    def feature_seq_lens(self) -> dict[str, int]:
        return {
            name: emb.seq_len for name, emb in self.embeddable_features() if emb.seq_len is not None
        }

    def op_source_kind(self) -> dict[str, str]:
        feat_kind: dict[str, str] = {}
        for name, src in self._sources.items():
            k = "user" if src.source in ("User", "Context") else "item"
            feat_kind[name] = k
        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            kinds = [feat_kind[inp] for inp in def_.inputs if inp in feat_kind]
            if any(k == "user" for k in kinds) and any(k == "item" for k in kinds):
                k = "cross"
            elif any(k == "user" for k in kinds):
                k = "user"
            elif any(k == "item" for k in kinds):
                k = "item"
            else:
                k = kinds[0] if kinds else "other"
            for out_name in def_.outputs:
                feat_kind[out_name] = k
        op_kind: dict[str, str] = {}
        for node_name in self._execution_order:
            def_ = self._node_defs[node_name]
            kinds = [feat_kind[inp] for inp in def_.inputs if inp in feat_kind]
            if any(k == "user" for k in kinds) and any(k == "item" for k in kinds):
                k = "cross"
            elif any(k == "user" for k in kinds):
                k = "user"
            elif any(k == "item" for k in kinds):
                k = "item"
            else:
                k = "other"
            op_kind[node_name] = k
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
