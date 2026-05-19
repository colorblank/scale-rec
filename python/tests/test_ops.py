from train.ops.bucketing import Bucketing
from train.ops.cross_feature import CrossFeature
from train.ops.dict_mapper import DictMapper
from train.ops.expression import ExpressionOp
from train.ops.sequence import SequenceOp
from train.ops.string_parser import StringParser
from train.ops.json_extract_list import JsonExtractList
from train.ops.list_string_parser import ListStringParser


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


def test_json_extract_list():
    op_obj = JsonExtractList("tag", 2, "none")
    result_obj = op_obj.process(['[{"score":0.99,"tag":"invest"}, {"score":0.5,"tag":"finance"}]'])
    assert result_obj == ["invest", "finance"]

    op_str = JsonExtractList(None, 2, "none")
    result_str = op_str.process(['["603538,17"]'])
    assert result_str == ["603538,17", "none"]


def test_list_string_parser():
    op = ListStringParser(",", 0)
    result = op.process([["603538,17", "000001,33"]])
    assert result == ["603538", "000001"]

    op2 = ListStringParser(",", 1)
    result2 = op2.process([["603538,17", "000001,33"]])
    assert result2 == ["17", "33"]


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
