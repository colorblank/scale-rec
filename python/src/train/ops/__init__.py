"""Feature operator library — mirrors src/feats/ops/."""

from __future__ import annotations

from typing import Any, Protocol

from ..core.config import OpType


class CustomOp(Protocol):
    def process(self, inputs: list[Any]) -> Any: ...


class OpFactory(Protocol):
    @classmethod
    def from_config(cls, params: dict) -> CustomOp: ...


# ── Operator registry ──

OP_REGISTRY: dict[str, OpFactory] = {}


def register_op(op_type: str):
    """Decorator that registers an operator class by its YAML op_type string."""

    def decorator(cls):
        OP_REGISTRY[op_type] = cls
        return cls

    return decorator


def create_op(op_type: str | OpType, params: dict) -> CustomOp:
    """Look up operator in registry and construct from params."""
    key = op_type.value if isinstance(op_type, OpType) else op_type
    cls = OP_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unsupported operator type: {key}")
    return cls.from_config(params)


# ── Operator imports (triggers registration via decorator) ──

from .bucketing import Bucketing as Bucketing
from .concat_hash import ConcatHash as ConcatHash
from .cross_feature import CrossFeature as CrossFeature
from .dict_mapper import DictMapper as DictMapper
from .expression import ExpressionOp as ExpressionOp
from .feature_hash import FeatureHash as FeatureHash
from .flat_split import FlatSplit as FlatSplit
from .json_extract_list import JsonExtractList as JsonExtractList
from .list_overlap import ListOverlap as ListOverlap
from .list_string_parser import ListStringParser as ListStringParser
from .log1p import Log1p as Log1p
from .parsed_feature_hash import ParsedFeatureHash as ParsedFeatureHash
from .sequence import SequenceOp as SequenceOp
from .split import Split as Split
from .string_concat import StringConcat as StringConcat
from .string_parser import StringParser as StringParser
from .time_parser import TimeParser as TimeParser

__all__ = [
    "OP_REGISTRY",
    "Bucketing",
    "ConcatHash",
    "CrossFeature",
    "CustomOp",
    "DictMapper",
    "ExpressionOp",
    "FeatureHash",
    "FlatSplit",
    "JsonExtractList",
    "ListOverlap",
    "ListStringParser",
    "Log1p",
    "ParsedFeatureHash",
    "SequenceOp",
    "Split",
    "StringConcat",
    "StringParser",
    "TimeParser",
    "create_op",
    "register_op",
]
