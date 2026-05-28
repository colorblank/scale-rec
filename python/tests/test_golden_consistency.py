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

def test_python_dag_matches_golden_fixture():
    dag = FeatureDag(FlowConfig.from_yaml(str(FIXTURE_DIR / "golden_feature_config.yaml")))
    rows = [
        {k: v for k, v in json.loads(line).items() if v is not None}
        for line in (FIXTURE_DIR / "golden_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = json.loads((FIXTURE_DIR / "golden_expected.json").read_text())

    actual = [
        {name: round_floats(result.features[name]) for name in expected_row}
        for result, expected_row in (
            (dag.execute(row), expected_row)
            for row, expected_row in zip(rows, expected, strict=True)
        )
    ]

    assert actual == expected
