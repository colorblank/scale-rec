from __future__ import annotations

"""StringConcat 算子测试。"""

from train.ops.string_concat import StringConcat


class TestStringConcatSingle:
    def test_single_input(self):
        op = StringConcat("_")
        assert op.process(["hello"]) == "hello"

    def test_empty_input(self):
        op = StringConcat("_")
        assert op.process([]) == ""

    def test_none_input(self):
        op = StringConcat("_")
        assert op.process([None]) == ""


class TestStringConcatMulti:
    def test_two_strings(self):
        op = StringConcat("_")
        assert op.process(["user", "123"]) == "user_123"

    def test_int_and_str(self):
        op = StringConcat("|")
        assert op.process([42, "abc"]) == "42|abc"

    def test_int_input(self):
        op = StringConcat("_")
        assert op.process([1, 2, 3]) == "1_2_3"

    def test_custom_separator(self):
        op = StringConcat("#")
        assert op.process(["a", "b", "c"]) == "a#b#c"


class TestStringConcatBatch:
    def test_batch_two_rows(self):
        op = StringConcat("_")
        result = op.process_batch([["x", "y"], ["1", "2"]])
        assert result == ["x_1", "y_2"]

    def test_batch_three_rows(self):
        op = StringConcat("|")
        result = op.process_batch([["a", "b", "c"], ["d", "e", "f"]])
        assert result == ["a|d", "b|e", "c|f"]

    def test_batch_none_values(self):
        op = StringConcat("_")
        result = op.process_batch([["a", None], ["b", "c"]])
        assert result == ["a_b", "_c"]

    def test_batch_empty(self):
        op = StringConcat("_")
        assert op.process_batch([]) == []
        assert op.process_batch([[]]) == []
