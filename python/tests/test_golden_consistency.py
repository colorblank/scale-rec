import json
from pathlib import Path

from train.config import FlowConfig
from train.dag import FeatureDag


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def test_python_dag_matches_golden_fixture():
    dag = FeatureDag(FlowConfig.from_yaml(str(FIXTURE_DIR / "golden_feature_config.yaml")))
    rows = [
        json.loads(line) for line in (FIXTURE_DIR / "golden_rows.jsonl").read_text().splitlines()
    ]
    expected = json.loads((FIXTURE_DIR / "golden_expected.json").read_text())

    actual = [
        {name: result.features[name] for name in expected_row}
        for result, expected_row in (
            (dag.execute(row), expected_row)
            for row, expected_row in zip(rows, expected, strict=True)
        )
    ]

    assert actual == expected
