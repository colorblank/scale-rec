from typing import Any, Protocol
class CustomOp(Protocol):
    def process(self, inputs: list[Any]) -> Any: ...
from .bucketing import Bucketing
from .cross_feature import CrossFeature
from .dict_mapper import DictMapper
from .expression import ExpressionOp
from .sequence import SequenceOp
from .string_parser import StringParser
