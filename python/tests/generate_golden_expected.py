import json
from pathlib import Path

from train.core.config import FlowConfig
from train.core.dag import FeatureDag

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def round_floats(val):
    if isinstance(val, float):
        return round(val, 5)
    if isinstance(val, list):
        return [round_floats(x) for x in val]
    if isinstance(val, dict):
        return {k: round_floats(v) for k, v in val.items()}
    return val


def main():
    dag = FeatureDag(FlowConfig.from_yaml(str(FIXTURE_DIR / "golden_feature_config.yaml")))

    rows = [
        {k: v for k, v in json.loads(line).items() if v is not None}
        for line in (FIXTURE_DIR / "golden_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]

    features_to_assert = [
        "age_bucket",
        "category_idx",
        "user_tag_keys",
        "item_tag_keys",
        "user_tag_idx",
        "tag_overlap",
        "tag_cross",
        "tag_cross_idx",
        "session_idx",
        "json_tags",
        "list_parsed",
        "split_list",
        "flat_split_list",
        "expr_out",
        "seq_out",
        "concat_out",
    ]

    expected = []
    for row in rows:
        result = dag.execute(row)
        expected_row = {name: round_floats(result.features[name]) for name in features_to_assert}
        expected.append(expected_row)

    (FIXTURE_DIR / "golden_expected.json").write_text(json.dumps(expected, indent=2))
    print(f"Successfully generated golden_expected.json with {len(expected)} rows.")


if __name__ == "__main__":
    main()
