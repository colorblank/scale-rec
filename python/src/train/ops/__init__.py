"""Feature operator library — mirrors src/feats/ops/."""

from __future__ import annotations

from typing import Any, Protocol


class CustomOp(Protocol):
    def process(self, inputs: list[Any]) -> Any: ...


# ── Operator registry ──

OP_REGISTRY: dict[str, type] = {}


def register_op(op_type: str):
    """Decorator that registers an operator class by its YAML op_type string."""
    def decorator(cls):
        OP_REGISTRY[op_type] = cls
        return cls
    return decorator


def create_op(op_type: str, params: dict) -> CustomOp:
    """Look up operator in registry and construct from params."""
    cls = OP_REGISTRY.get(op_type)
    if cls is None:
        raise ValueError(f"Unsupported operator type: {op_type}")
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
from .parsed_feature_hash import ParsedFeatureHash as ParsedFeatureHash
from .sequence import SequenceOp as SequenceOp
from .split import Split as Split
from .string_concat import StringConcat as StringConcat
from .string_parser import StringParser as StringParser

__all__ = [
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
    "ParsedFeatureHash",
    "SequenceOp",
    "Split",
    "StringConcat",
    "StringParser",
    "OP_REGISTRY",
    "create_op",
    "register_op",
]
