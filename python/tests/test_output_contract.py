from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from train.core.output_contract import SourceContract, parse_output_contract

CASES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "output_contract_cases.yaml"
CANONICAL = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "output_contract_canonical.json"
)


@pytest.mark.parametrize("case", yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"])
def test_output_contract_shared_cases(case):
    sources = [SourceContract(**source) for source in case["sources"]]
    if case["valid"]:
        contract = parse_output_contract(case["contract"], sources)
        assert contract.version == 1
        assert contract.node_kinds["ctcvr_prob"] == "probability"
        assert contract.relation_order == ("click_prob", "cvr_prob", "ctcvr_prob")
    else:
        with pytest.raises((TypeError, ValueError), match=case["error"]):
            parse_output_contract(case["contract"], sources)


def test_output_contract_mask_must_reference_data_source():
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"][0]["contract"]
    raw["objectives"][0]["mask"] = {"source": "missing", "op": "eq", "value": 1}
    sources = [
        SourceContract("is_click", "label"),
        SourceContract("is_conversion", "label"),
    ]

    with pytest.raises(ValueError, match="mask source"):
        parse_output_contract(raw, sources)


def test_canonical_json_ignores_declaration_order():
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"][0]["contract"]
    reordered = yaml.safe_load(yaml.safe_dump(raw))
    reordered["graph"]["towers"].reverse()
    reordered["graph"]["relations"].reverse()
    reordered["objectives"].reverse()
    reordered["metrics"].reverse()
    reordered["outputs"].reverse()

    expected = CANONICAL.read_text(encoding="utf-8").strip()
    assert parse_output_contract(raw).canonical_json() == expected
    assert parse_output_contract(reordered).canonical_json() == expected
