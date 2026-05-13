from train.ops.bucketing import Bucketing
from train.ops.cross_feature import CrossFeature
from train.ops.dict_mapper import DictMapper
from train.ops.expression import ExpressionOp
from train.ops.sequence import SequenceOp
from train.ops.string_parser import StringParser


def test_bucketing():
    op = Bucketing([18.0, 25.0, 35.0, 50.0])
    assert op.process([28.5]) == 2
    assert op.process([5.0]) == 0
    assert op.process([60.0]) == 4


def test_dict_mapper():
    op = DictMapper({"elec": 1, "book": 2}, default_idx=0)
    assert op.process(["elec"]) == 1
    assert op.process(["unknown"]) == 0


def test_string_parser():
    op = StringParser("|", "#", 0, 2, "none")
    result = op.process(["sports#1|gaming#0.8"])
    assert result == ["sports", "gaming"]


def test_expression():
    op = ExpressionOp("log(v0 + 1.0)")
    result = op.process([5999.0])
    assert abs(result - 8.6995) < 0.01


def test_sequence():
    op = SequenceOp(3, 0)
    assert op.process([[1, 2]]) == [1, 2, 0]
    assert op.process([[1, 2, 3, 4]]) == [1, 2, 3]


def test_cross_feature_inner_product():
    op = CrossFeature("inner_product")
    assert op.process([[1.0, 2.0], [3.0, 4.0]]) == 11.0
