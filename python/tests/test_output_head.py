from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from train.core.model_output import ModelOutput
from train.core.output_contract import parse_output_contract
from train.models.output_head import OutputHead
from train.training.loss.objective import ObjectiveEngine

CASES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "output_contract_cases.yaml"


def _esmm_contract():
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"][0]["contract"]
    return parse_output_contract(raw)


def _zero_head() -> OutputHead:
    head = OutputHead(_esmm_contract(), {"shared": 4})
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    return head


def _single_objective_contract(kind: str, loss: dict):
    return parse_output_contract(
        {
            "version": 1,
            "graph": {
                "towers": [{"name": "prediction", "kind": kind}],
                "relations": [],
            },
            "objectives": [
                {
                    "name": "loss",
                    "source": "prediction",
                    "label": "label",
                    "loss": loss,
                }
            ],
            "metrics": [],
            "outputs": [{"name": "prediction", "source": "prediction"}],
        }
    )


def test_output_head_executes_esmm_probability_graph_and_projection():
    execution = _zero_head()({"shared": torch.zeros(2, 4)})

    assert execution.nodes.kind("click_logit") == "binary_logit"
    assert torch.allclose(execution.nodes.tensor("click_prob"), torch.full((2, 1), 0.5))
    assert torch.allclose(execution.nodes.tensor("ctcvr_prob"), torch.full((2, 1), 0.25))
    assert execution.outputs.names() == ["ctr", "ctcvr"]
    assert execution.outputs.kind("ctcvr") == "probability"
    assert "click_logit" not in execution.outputs


def test_output_head_weight_names_match_candle_paths():
    keys = set(_zero_head().state_dict())

    assert "towers.click_logit.hidden.0.weight" in keys
    assert "towers.click_logit.output.2.weight" in keys
    assert "towers.cvr_logit.output.2.bias" in keys


def test_output_head_executes_regression_add_and_identity():
    contract = parse_output_contract(
        {
            "version": 1,
            "graph": {
                "towers": [
                    {"name": "left", "kind": "regression"},
                    {"name": "right", "kind": "regression"},
                ],
                "relations": [
                    {"name": "sum", "op": "add", "inputs": ["left", "right"]},
                    {"name": "value", "op": "identity", "inputs": ["sum"]},
                ],
            },
            "objectives": [],
            "metrics": [],
            "outputs": [{"name": "prediction", "source": "value"}],
        }
    )
    head = OutputHead(contract, {"shared": 2})
    execution = head({"shared": torch.zeros(1, 2)})

    assert torch.equal(
        execution.outputs.tensor("prediction"),
        execution.nodes.tensor("left") + execution.nodes.tensor("right"),
    )
    assert execution.outputs.kind("prediction") == "regression"


def test_objective_engine_uses_probability_focal_after_esmm_relation():
    execution = _zero_head()({"shared": torch.zeros(2, 4)})

    result = ObjectiveEngine(_esmm_contract())(
        execution,
        {
            "is_click": [0, 1],
            "is_conversion": [1, 1],
        },
    )

    assert result.total is not None
    assert torch.allclose(result.losses["click_loss"], torch.tensor(0.6931472))
    assert torch.allclose(result.losses["conversion_loss"], torch.tensor(0.1949476), atol=1e-6)
    assert torch.allclose(result.total, torch.tensor(0.8880948), atol=1e-6)


def test_objective_engine_rejects_runtime_kind_mismatch():
    nodes = ModelOutput()
    nodes.insert_binary_logit("click_logit", torch.zeros(1, 1))
    nodes.insert_binary_logit("ctcvr_prob", torch.zeros(1, 1))

    with pytest.raises(ValueError, match="expected probability"):
        ObjectiveEngine(_esmm_contract())(
            nodes,
            {
                "is_click": [1],
                "is_conversion": [1],
            },
        )


def test_objective_engine_supports_bce_pos_weight():
    contract = _single_objective_contract(
        "binary_logit",
        {"type": "binary_cross_entropy_with_logits", "pos_weight": 2.0},
    )
    nodes = ModelOutput()
    nodes.insert_binary_logit("prediction", torch.zeros(1, 1))

    result = ObjectiveEngine(contract)(nodes, {"label": [1]})

    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(1.3862944))


def test_objective_engine_supports_focal_bce_with_logits():
    contract = _single_objective_contract(
        "binary_logit",
        {"type": "focal_binary_cross_entropy_with_logits"},
    )
    nodes = ModelOutput()
    nodes.insert_binary_logit("prediction", torch.zeros(1, 1))

    result = ObjectiveEngine(contract)(nodes, {"label": [1]})

    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(0.1732868), atol=1e-6)


def test_objective_engine_clamps_probability_bce():
    contract = parse_output_contract(
        {
            "version": 1,
            "graph": {
                "towers": [{"name": "logit", "kind": "binary_logit"}],
                "relations": [{"name": "prediction", "op": "sigmoid", "inputs": ["logit"]}],
            },
            "objectives": [
                {
                    "name": "loss",
                    "source": "prediction",
                    "label": "label",
                    "loss": {"type": "binary_cross_entropy", "epsilon": 1e-4},
                }
            ],
            "metrics": [],
            "outputs": [{"name": "prediction", "source": "prediction"}],
        }
    )
    nodes = ModelOutput()
    nodes.insert_probability("prediction", torch.zeros(1, 1))

    result = ObjectiveEngine(contract)(nodes, {"label": [1]})

    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(9.2103404), atol=1e-6)


@pytest.mark.parametrize(
    ("loss_type", "expected"),
    [
        ("mse", 2.5),
        ("mae", 1.5),
        ("huber", 1.0),
    ],
)
def test_objective_engine_regression_losses(loss_type, expected):
    loss = {"type": loss_type}
    if loss_type == "huber":
        loss["delta"] = 1.0
    contract = _single_objective_contract("regression", loss)
    nodes = ModelOutput()
    nodes.insert_regression("prediction", torch.tensor([[1.0], [3.0]]))

    result = ObjectiveEngine(contract)(nodes, {"label": [2.0, 1.0]})

    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(expected))


def test_objective_engine_weighted_stay_loss():
    contract = _single_objective_contract("binary_logit", {"type": "weighted_bce_stay"})
    nodes = ModelOutput()
    nodes.insert_binary_logit("prediction", torch.zeros(2, 1))

    result = ObjectiveEngine(contract)(nodes, {"label": [0.0, 10.0]})

    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(0.6931472))


def test_objective_engine_applies_structured_mask_before_reduction():
    raw = yaml.safe_load(CASES.read_text(encoding="utf-8"))["cases"][0]["contract"]
    raw["objectives"] = [
        {
            "name": "masked_click",
            "source": "click_logit",
            "label": "is_click",
            "loss": {"type": "binary_cross_entropy_with_logits", "reduction": "sum"},
            "mask": {"source": "eligible", "op": "eq", "value": 1},
        }
    ]
    contract = parse_output_contract(raw)
    execution = _zero_head()({"shared": torch.zeros(2, 4)})

    result = ObjectiveEngine(contract)(
        execution,
        {
            "is_click": [1, 0],
            "eligible": [1, 0],
        },
    )

    assert result.sample_counts == {"masked_click": 1}
    assert result.total is not None
    assert torch.allclose(result.total, torch.tensor(0.6931472))


def test_output_head_rejects_unknown_backbone_representation():
    with pytest.raises(ValueError, match="unknown representation 'shared'"):
        OutputHead(_esmm_contract(), {})
