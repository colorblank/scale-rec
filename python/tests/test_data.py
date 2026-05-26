from __future__ import annotations

from pathlib import Path

from train.app.data import build_item_index


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
