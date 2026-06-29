import pytest

from train.ops.bucketing import Bucketing
from train.ops.cross_feature import CrossFeature
from train.ops.dict_mapper import DictMapper
from train.ops.expression import ExpressionOp
from train.ops.flat_split import FlatSplit
from train.ops.json_extract_list import JsonExtractList
from train.ops.list_string_parser import ListStringParser
from train.ops.log1p import Log1p
from train.ops.sequence import SequenceOp
from train.ops.split import Split
from train.ops.string_parser import StringParser
from train.ops.time_parser import TimeParser


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


def test_log1p():
    op = Log1p()
    result = op.process([5999.0])
    assert abs(result - 8.699515) < 1e-6
    assert op.process([0]) == 0.0


def test_log1p_rejects_domain_error():
    op = Log1p()
    with pytest.raises(ValueError, match="greater than -1"):
        op.process([-1.0])


def test_time_parser_rfc3339_to_utc_hour():
    op = TimeParser(input_format="rfc3339", output="hour", default_val=-1)
    assert op.process(["2026-06-29T10:30:00+08:00"]) == 2


def test_time_parser_custom_format_to_weekday():
    op = TimeParser(
        input_format="strftime",
        output="weekday",
        formats=["%Y.%m.%d %H:%M"],
        default_val=-1,
    )
    assert op.process(["2026.06.29 12:00"]) == 0


def test_time_parser_epoch_ms_to_yyyymmdd():
    op = TimeParser(input_format="epoch_ms", output="yyyymmdd", default_val=-1)
    assert op.process(["1782727200000"]) == 20260629


def test_time_parser_invalid_uses_default():
    op = TimeParser(input_format="auto", output="day_of_year", default_val=366)
    assert op.process(["bad"]) == 366


def test_sequence():
    op = SequenceOp(3, 0)
    assert op.process([[1, 2]]) == [1, 2, 0]
    assert op.process([[1, 2, 3, 4]]) == [1, 2, 3]


def test_cross_feature_inner_product():
    op = CrossFeature("inner_product")
    assert op.process([[1.0, 2.0], [3.0, 4.0]]) == 11.0


def test_split():
    op = Split("|", 3, "none")
    assert op.process(["a|b|c"]) == ["a", "b", "c"]
    assert op.process(["a|b"]) == ["a", "b", "none"]
    assert op.process(["a|b|c|d"]) == ["a", "b", "c"]


def test_split_no_limit():
    op = Split("|", 0, "")
    assert op.process(["a|b|c"]) == ["a", "b", "c"]
    assert op.process(["hello"]) == ["hello"]


def test_split_cache_returns_independent_lists():
    op = Split("|", 3, "none")
    first = op.process(["中文|a"])
    first[0] = "mutated"

    assert op.process(["中文|a"]) == ["中文", "a", "none"]


def test_flat_split():
    op = FlatSplit(",", 8, "")
    result = op.process([["a_93,b_129,c_140,d_53", "a_51,b_245,c_205,d_157"]])
    assert result == ["a_93", "b_129", "c_140", "d_53", "a_51", "b_245", "c_205", "d_157"]


def test_flat_split_truncate():
    op = FlatSplit(",", 3, "none")
    result = op.process([["a,b,c,d"]])
    assert result == ["a", "b", "c"]


def test_flat_split_pad():
    op = FlatSplit(",", 4, "pad")
    result = op.process([["x"]])
    assert result == ["x", "pad", "pad", "pad"]
