"""Feature operator library — mirrors src/feats/ops/."""

from __future__ import annotations

from typing import Any, Protocol


class CustomOp(Protocol):
    def process(self, inputs: list[Any]) -> Any: ...


from .bucketing import Bucketing as Bucketing
from .cross_feature import CrossFeature as CrossFeature
from .dict_mapper import DictMapper as DictMapper
from .expression import ExpressionOp as ExpressionOp
from .list_overlap import ListOverlap as ListOverlap
from .sequence import SequenceOp as SequenceOp
from .string_concat_hash import StringConcatHash as StringConcatHash
from .string_parser import StringParser as StringParser

__all__ = [
    "Bucketing",
    "CrossFeature",
    "CustomOp",
    "DictMapper",
    "ExpressionOp",
    "ListOverlap",
    "SequenceOp",
    "StringConcatHash",
    "StringParser",
]
