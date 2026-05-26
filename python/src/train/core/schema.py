from __future__ import annotations

"""Static feature schema inference and validation for FlowConfig."""

from dataclasses import dataclass

from .config import (
    DType,
    EmbedConfig,
    FlowConfig,
    OperatorDef,
    Role,
    parse_float_strict,
    parse_int_strict,
)


@dataclass(frozen=True)
class FeatureDType:
    tag: str
    inner: "FeatureDType | None" = None
    length: int | None = None

    @property
    def is_list(self) -> bool:
        return self.tag == "list"

    @property
    def is_integer_index(self) -> bool:
        return self.tag == "int" or (
            self.tag == "list" and self.inner is not None and self.inner.tag == "int"
        )

    def __str__(self) -> str:
        if self.tag == "list" and self.inner is not None:
            length = "" if self.length is None else f";{self.length}"
            return f"list[{self.inner}{length}]"
        return self.tag


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    dtype: FeatureDType
    rank: int
    nullable: bool = False
    default_val: str | None = None
    cardinality: int | None = None
    pooling: str | None = None


def infer_feature_schemas(config: FlowConfig) -> dict[str, FeatureSchema]:
    """Infer all source/operator output schemas and validate embeddable features."""
    schemas: dict[str, FeatureSchema] = {}
    for source in config.sources:
        if source.role != Role.FEATURE:
            continue
        dtype = _dtype_from_config(source.dtype)
        _validate_default(source.name, dtype, source.default_val)
        schemas[source.name] = FeatureSchema(
            name=source.name,
            dtype=dtype,
            rank=1 if dtype.is_list else 0,
            nullable=False,
            default_val=source.default_val,
        )
        if source.embed is not None:
            _validate_embed(source.name, dtype, source.embed)

    for op in config.operators:
        input_schemas = [_require_schema(schemas, op.name, name) for name in op.inputs]
        output_schema = _infer_operator_output(op, input_schemas)
        for output in op.outputs:
            dtype = output_schema.dtype
            if op.embed is not None:
                _validate_embed(output, dtype, op.embed)
                output_schema = FeatureSchema(
                    name=output,
                    dtype=dtype,
                    rank=output_schema.rank,
                    nullable=output_schema.nullable,
                    default_val=output_schema.default_val,
                    cardinality=op.embed.vocab_size,
                    pooling=op.embed.pooling,
                )
            else:
                output_schema = FeatureSchema(
                    name=output,
                    dtype=dtype,
                    rank=output_schema.rank,
                    nullable=output_schema.nullable,
                    default_val=output_schema.default_val,
                    cardinality=output_schema.cardinality,
                    pooling=output_schema.pooling,
                )
            schemas[output] = output_schema
    return schemas


def _dtype_from_config(dtype: DType) -> FeatureDType:
    if dtype.tag == "list":
        if dtype.inner is None:
            raise ValueError("list dtype requires inner dtype")
        return FeatureDType("list", _dtype_from_config(dtype.inner), dtype.length)
    return FeatureDType(dtype.tag)


def _validate_default(name: str, dtype: FeatureDType, default_val: str) -> None:
    try:
        if dtype.tag == "int":
            parse_int_strict(default_val)
        elif dtype.tag == "float":
            parse_float_strict(default_val)
        elif dtype.tag == "list" and dtype.inner is not None:
            _validate_default(name, dtype.inner, default_val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source '{name}' default '{default_val}' does not match dtype {dtype}"
        ) from exc


def _require_schema(
    schemas: dict[str, FeatureSchema], op_name: str, feature_name: str
) -> FeatureSchema:
    if feature_name not in schemas:
        raise ValueError(f"operator '{op_name}' references unknown input '{feature_name}'")
    return schemas[feature_name]


def _infer_operator_output(op: OperatorDef, input_schemas: list[FeatureSchema]) -> FeatureSchema:
    first = input_schemas[0] if input_schemas else None
    op_type = op.op_type
    params = op.params
    if op_type == "Bucketing":
        _require_scalar_number(op, first)
        return _schema(op, FeatureDType("int"))
    if op_type == "DictMapper":
        if first and first.dtype.is_list:
            return _schema(op, FeatureDType("list", FeatureDType("int"), first.dtype.length))
        return _schema(op, FeatureDType("int"))
    if op_type in {"StringParser", "JsonExtractList"}:
        return _schema(
            op, FeatureDType("list", FeatureDType("string"), int(params.get("pad_len", 0)) or None)
        )
    if op_type == "ListStringParser":
        length = first.dtype.length if first and first.dtype.is_list else None
        return _schema(op, FeatureDType("list", FeatureDType("string"), length))
    if op_type in {"Split", "FlatSplit"}:
        return _schema(
            op, FeatureDType("list", FeatureDType("string"), int(params.get("max_len", 0)) or None)
        )
    if op_type == "CrossFeature":
        if params.get("cross_type") == "inner_product":
            return _schema(op, FeatureDType("float"))
        return _schema(op, FeatureDType("list", FeatureDType("string"), None))
    if op_type == "ExpressionOp":
        return _schema(op, FeatureDType("float"))
    if op_type == "SequenceOp":
        return _schema(
            op, FeatureDType("list", FeatureDType("int"), int(params.get("max_len", 10)))
        )
    if op_type == "ListOverlap":
        return _schema(op, FeatureDType("int"))
    if op_type == "StringConcat":
        return _schema(op, FeatureDType("string"))
    if op_type == "FeatureHash":
        has_list_input = any(s.dtype.is_list for s in input_schemas)
        num_hashes = int(params.get("num_hashes", 1))
        if has_list_input:
            length = next((s.dtype.length for s in input_schemas if s.dtype.is_list), None)
            return _schema(op, FeatureDType("list", FeatureDType("int"), length))
        if num_hashes > 1:
            return _schema(op, FeatureDType("list", FeatureDType("int"), num_hashes))
        return _schema(op, FeatureDType("int"))
    if op_type == "PluginOp":
        return _schema(op, FeatureDType("unknown"))
    raise ValueError(f"Unsupported operator for schema inference: {op_type}")


def _schema(op: OperatorDef, dtype: FeatureDType) -> FeatureSchema:
    return FeatureSchema(
        name=op.outputs[0] if op.outputs else op.name, dtype=dtype, rank=1 if dtype.is_list else 0
    )


def _require_scalar_number(op: OperatorDef, schema: FeatureSchema | None) -> None:
    if schema is None or schema.dtype.tag not in {"int", "float"}:
        got = "missing" if schema is None else str(schema.dtype)
        raise ValueError(f"operator '{op.name}' expects numeric scalar input, got {got}")


def _validate_embed(name: str, dtype: FeatureDType, embed: EmbedConfig) -> None:
    if embed.vocab_size <= 0:
        raise ValueError(f"embed '{name}' vocab_size must be positive")
    if embed.embed_dim <= 0:
        raise ValueError(f"embed '{name}' embed_dim must be positive")
    if embed.pooling not in {"first", "mean", "sum", "max", "flatten"}:
        raise ValueError(f"embed '{name}' has unsupported pooling '{embed.pooling}'")
    if not dtype.is_integer_index:
        raise ValueError(f"embeddable feature '{name}' must be int or list[int], got {dtype}")
    if embed.pooling in {"mean", "sum", "max", "flatten"} and not dtype.is_list:
        raise ValueError(f"embed '{name}' pooling '{embed.pooling}' requires list[int]")
    if embed.pooling == "flatten":
        seq_len = embed.seq_len or dtype.length
        if not seq_len:
            raise ValueError(f"embed '{name}' pooling flatten requires seq_len")
