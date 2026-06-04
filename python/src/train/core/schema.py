from __future__ import annotations

"""Static feature schema inference and validation for FlowConfig."""
from typing import Optional

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
    inner: "Optional[FeatureDType]" = None
    length: Optional[int] = None
    values: Optional[tuple[str, ...]] = None
    default: Optional[str] = None
    oov: Optional[str] = None

    @property
    def is_list(self) -> bool:
        return self.tag == "list"

    @property
    def is_integer_index(self) -> bool:
        return self.tag == "int" or (
            self.tag == "list" and self.inner is not None and self.inner.tag == "int"
        )

    @property
    def dimension(self) -> int:
        return self.length or 1 if self.is_list else 1

    def __str__(self) -> str:
        if self.tag == "list" and self.inner is not None:
            length = "" if self.length is None else f";{self.length}"
            return f"list[{self.inner}{length}]"
        if self.tag == "enum":
            return "enum"
        return self.tag


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    dtype: FeatureDType
    rank: int
    dimension: int
    nullable: bool = False
    default_val: Optional[str] = None
    cardinality: Optional[int] = None
    pooling: Optional[str] = None


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
            dimension=dtype.dimension,
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
                    dimension=_embedding_dimension(dtype, op.embed),
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
                    dimension=output_schema.dimension,
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
        if not dtype.max_len or dtype.max_len <= 0:
            raise ValueError("list dtype requires max_len > 0")
        return FeatureDType("list", _dtype_from_config(dtype.inner), dtype.max_len)
    if dtype.tag == "enum":
        values = tuple(dtype.values or ())
        if not values:
            raise ValueError("enum dtype requires values")
        return FeatureDType("enum", values=values, default=dtype.default, oov=dtype.oov)
    return FeatureDType(dtype.tag)


def _validate_default(name: str, dtype: FeatureDType, default_val: str) -> None:
    try:
        if dtype.tag == "int":
            parse_int_strict(default_val)
        elif dtype.tag == "float":
            parse_float_strict(default_val)
        elif dtype.tag == "enum":
            if dtype.values and default_val not in dtype.values and default_val != dtype.oov:
                raise ValueError(f"unknown enum value '{default_val}'")
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
        if len(input_schemas) != 2:
            raise ValueError(f"operator '{op.name}' expects exactly 2 inputs")
        if params.get("cross_type") == "inner_product":
            return _schema(op, FeatureDType("float"))
        max_len = params.get("max_len")
        if max_len is not None:
            length = int(max_len)
        else:
            lengths = [schema.dtype.length for schema in input_schemas if schema.dtype.is_list]
            length = 1
            for item in lengths:
                if item is None:
                    length = 0
                    break
                length *= item
            if not lengths or length <= 0:
                length = None
        return _schema(op, FeatureDType("list", FeatureDType("string"), length))
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
        list_inputs = [s for s in input_schemas if s.dtype.is_list]
        num_hashes = int(params.get("num_hashes", 1))
        if list_inputs:
            length = 0
            for schema in input_schemas:
                if schema.dtype.is_list:
                    if schema.dtype.length is None:
                        length = None
                        break
                    length += schema.dtype.length
                else:
                    length += 1
            return _schema(op, FeatureDType("list", FeatureDType("int"), length))
        if num_hashes > 1:
            return _schema(op, FeatureDType("list", FeatureDType("int"), num_hashes))
        return _schema(op, FeatureDType("int"))
    if op_type == "PluginOp":
        return _schema(op, FeatureDType("unknown"))
    raise ValueError(f"Unsupported operator for schema inference: {op_type}")


def _schema(op: OperatorDef, dtype: FeatureDType) -> FeatureSchema:
    return FeatureSchema(
        name=op.outputs[0] if op.outputs else op.name,
        dtype=dtype,
        rank=1 if dtype.is_list else 0,
        dimension=dtype.dimension,
    )


def _require_scalar_number(op: OperatorDef, schema: Optional[FeatureSchema]) -> None:
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
    if dtype.is_list:
        seq_len = embed.seq_len or dtype.length
        if not seq_len:
            raise ValueError(f"embed '{name}' list input requires fixed max_len or seq_len")


def _embedding_dimension(dtype: FeatureDType, embed: EmbedConfig) -> int:
    if dtype.is_list:
        return embed.seq_len or dtype.length or 1
    return 1
