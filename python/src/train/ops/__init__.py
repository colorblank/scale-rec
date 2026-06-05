"""Feature operator library — mirrors src/feats/ops/."""

from __future__ import annotations

from typing import Any, Protocol


class CustomOp(Protocol):
    def process(self, inputs: list[Any]) -> Any: ...


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
    "CrossFeature",
    "CustomOp",
    "DictMapper",
    "ExpressionOp",
    "FeatureHash",
    "ConcatHash",
    "FlatSplit",
    "ListOverlap",
    "SequenceOp",
    "Split",
    "StringConcat",
    "StringParser",
    "JsonExtractList",
    "ListStringParser",
    "ParsedFeatureHash",
]
