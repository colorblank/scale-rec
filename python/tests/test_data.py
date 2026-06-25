from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from train.app.cli import resolve_data_paths
from train.app.data import (
    build_item_index,
    estimate_files_batches,
    stream_files_batches,
    validate_matching_text_format,
)
from train.core.config import FlowConfig


def test_build_item_index_honors_null_markers(tmp_path: Path):
    item_file = tmp_path / "items.tsv"
    item_file.write_text("item_id\tcategory\n1\tNULL\n2\tbooks\n", encoding="utf-8")

    index = build_item_index(
        [str(item_file)],
        [
            {"name": "item_id", "dtype": "string", "default_val": "", "role": "feature"},
            {"name": "category", "dtype": "string", "default_val": "", "role": "feature"},
        ],
        has_header=True,
        separator="\t",
        null_markers={"NULL"},
    )

    assert index["1"]["category"] == ""
    assert index["2"]["category"] == "books"


def test_resolve_data_paths_expands_date_glob_inclusive_sorted(tmp_path: Path):
    (tmp_path / "user_20260326.txt").write_text("d26\n", encoding="utf-8")
    (tmp_path / "user_20260325.txt").write_text("d25\n", encoding="utf-8")
    (tmp_path / "user_20260327.txt").write_text("d27\n", encoding="utf-8")
    (tmp_path / "user_latest.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "user_20260328.txt").write_text("outside\n", encoding="utf-8")

    args = argparse.Namespace(
        data=None,
        data_glob=str(tmp_path / "user_*.txt"),
        start_date="20260325",
        end_date="20260327",
    )

    assert [Path(p).name for p in resolve_data_paths(args)] == [
        "user_20260325.txt",
        "user_20260326.txt",
        "user_20260327.txt",
    ]


def test_resolve_data_paths_reports_missing_dates(tmp_path: Path):
    (tmp_path / "user_20260325.txt").write_text("d25\n", encoding="utf-8")
    (tmp_path / "user_20260327.txt").write_text("d27\n", encoding="utf-8")

    args = argparse.Namespace(
        data="fallback.csv",
        data_glob=str(tmp_path / "user_*.txt"),
        start_date="20260325",
        end_date="20260327",
    )

    with pytest.raises(SystemExit, match="20260326"):
        resolve_data_paths(args)


def test_stream_files_batches_yields_multiple_files_in_order(tmp_path: Path):
    first = tmp_path / "user_20260325.tsv"
    second = tmp_path / "user_20260326.tsv"
    first.write_text("user_id\tis_click\nu1\t1\nu2\t0\n", encoding="utf-8")
    second.write_text("user_id\tis_click\nu3\t1\n", encoding="utf-8")
    flow_config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": "", "role": "feature"},
                {"name": "is_click", "dtype": "int", "default_val": "0", "role": "label"},
            ],
            "operators": [],
        }
    )

    batches = list(stream_files_batches([str(first), str(second)], flow_config, 2, has_header=True))

    assert [batch["features"]["user_id"] for batch in batches] == [["u1", "u2"], ["u3"]]
    assert [batch["labels"]["is_click"] for batch in batches] == [[1, 0], [1]]


def test_validate_matching_text_format_requires_same_header_order(tmp_path: Path):
    training = tmp_path / "train.tsv"
    evaluation = tmp_path / "eval.tsv"
    training.write_text("user_id\tis_click\nu1\t1\n", encoding="utf-8")
    evaluation.write_text("is_click\tuser_id\n0\tu2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="columns must exactly match"):
        validate_matching_text_format(
            str(training),
            str(evaluation),
            has_header=True,
            sep="\t",
        )


def test_validate_matching_text_format_requires_same_width_without_header(tmp_path: Path):
    training = tmp_path / "train.tsv"
    evaluation = tmp_path / "eval.tsv"
    training.write_text("u1\t1\n", encoding="utf-8")
    evaluation.write_text("u2\t0\textra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column count must match"):
        validate_matching_text_format(
            str(training),
            str(evaluation),
            has_header=False,
            sep="\t",
        )


def test_stream_files_batches_usecols_skips_discard_without_header(tmp_path: Path):
    path = tmp_path / "wide.tsv"
    path.write_text("u1\tunused payload\t1\nu2\tmore unused\t0\n", encoding="utf-8")
    flow_config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": "", "role": "feature"},
                {"name": "payload", "dtype": "string", "default_val": "", "role": "discard"},
                {"name": "is_click", "dtype": "int", "default_val": "0", "role": "label"},
            ],
            "operators": [],
        }
    )

    batches = list(
        stream_files_batches(
            [str(path)],
            flow_config,
            1,
            has_header=False,
            read_chunk_rows=1,
            fast_no_na=True,
        )
    )

    assert [batch["features"] for batch in batches] == [{"user_id": ["u1"]}, {"user_id": ["u2"]}]
    assert [batch["labels"] for batch in batches] == [{"is_click": [1]}, {"is_click": [0]}]


def test_estimate_files_batches_matches_per_file_streaming(tmp_path: Path):
    first = tmp_path / "user_20260325.tsv"
    second = tmp_path / "user_20260326.tsv"
    first.write_text("user_id\tis_click\nu1\t1\nu2\t0\nu3\t1\n", encoding="utf-8")
    second.write_text("user_id\tis_click\nu4\t0\n", encoding="utf-8")

    assert estimate_files_batches([str(first), str(second)], 2, has_header=True) == 3


def test_read_chunk_rows_does_not_shrink_training_batch(tmp_path: Path):
    path = tmp_path / "train.tsv"
    path.write_text("user_id\tis_click\nu1\t1\nu2\t0\nu3\t1\n", encoding="utf-8")
    flow_config = FlowConfig.from_dict(
        {
            "version": "1.0.0",
            "sources": [
                {"name": "user_id", "dtype": "string", "default_val": "", "role": "feature"},
                {"name": "is_click", "dtype": "int", "default_val": "0", "role": "label"},
            ],
            "operators": [],
        }
    )

    batches = list(
        stream_files_batches(
            [str(path)],
            flow_config,
            2,
            has_header=True,
            read_chunk_rows=1,
        )
    )

    assert [batch["features"]["user_id"] for batch in batches] == [["u1", "u2"], ["u3"]]
