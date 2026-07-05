from pathlib import Path

import pytest
import torch

from train.core.config import ModelConfig
from train.layers.embedding import FeatureEmbeddings
from train.layers.gdcn import GatedCrossNetwork
from train.layers.towers import Activation, MultiTaskConfig, TaskRelation, TowerConfig
from train.models import get_output_spec
from train.models.deepfm import DeepFM
from train.models.esmm import ESMM
from train.models.gdcn_esmm import GDCNESMM
from train.models.lr import LogisticRegression
from train.models.mmoe import MMoE
from train.models.output import ModelOutput
from train.models.pepnet import GateNU, PEPNet
from train.models.rankup import RankUpConfig, RankUpModel

FEATURES = [("a", 10, 4), ("b", 5, 4)]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _inputs(batch=3):
    return {
        "a": torch.tensor([1, 2, 3][:batch]),
        "b": torch.tensor([0, 1, 2][:batch]),
    }


def test_lr_forward():
    model = LogisticRegression(FEATURES)
    out = model(_inputs(3))
    assert out.tensor("pred").shape == (3, 1)


def test_output_spec_accepts_task_specs():
    spec = get_output_spec(
        "lr",
        params={
            "tasks": [
                {
                    "name": "pred",
                    "label": "clicked",
                    "loss": "bce",
                    "weight": 0.5,
                    "metrics": ["auc", "logloss"],
                }
            ]
        },
    )

    assert spec["task_names"] == ["pred"]
    assert spec["label_col_map"] == {"pred": "clicked"}
    assert spec["output_kinds"] == {"pred": "binary_logit"}
    assert spec["tasks"][0].weight == 0.5


def test_output_spec_accepts_regression_task_specs():
    spec = get_output_spec(
        "lr",
        params={
            "tasks": [
                {
                    "name": "watch_time",
                    "label": "watch_time",
                    "loss": "huber",
                    "metrics": ["mae", "mse"],
                }
            ]
        },
    )

    assert spec["task_names"] == ["watch_time"]
    assert spec["output_kinds"] == {"watch_time": "regression"}
    assert spec["tasks"][0].output_kind == "regression"


def test_native_output_contract_esmm_exposes_public_and_internal_outputs():
    config = ModelConfig.from_yaml(REPO_ROOT / "examples/models/esmm_output_contract.yaml")
    model = config.build(FEATURES)
    spec = get_output_spec(config.type, model, config.params)

    public = model(_inputs(2))
    execution = model.forward_execution(_inputs(2))

    assert public.names() == ["ctr", "ctcvr", "ctdetail", "ctstock", "ctstay"]
    assert "click_logit" not in public
    assert execution.nodes.kind("click_logit") == "binary_logit"
    assert execution.nodes.kind("ctcvr_prob") == "probability"
    assert spec["label_col_map"]["ctcvr_prob"] == "is_cvr"
    assert spec["label_col_map"]["stay_logit"] == "stay_time_label"
    assert spec["task_metrics"]["ctcvr_prob"] == ["auc", "logloss"]
    assert "output_head.towers.click_logit.hidden.0.weight" in model.state_dict()


@pytest.mark.parametrize(
    "config_name",
    [
        "lr.yaml",
        "deepfm.yaml",
        "mmoe.yaml",
        "esmm_output_contract.yaml",
        "gdcn_esmm.yaml",
        "unimixer.yaml",
        "token_mixer_large.yaml",
        "rankmixer.yaml",
        "rankup.yaml",
        "hyformer.yaml",
        "pepnet.yaml",
    ],
)
def test_all_example_models_use_native_output_contract(config_name):
    from train.models.unimixer.tokenizer import FeatureTokenizer

    config = ModelConfig.from_yaml(REPO_ROOT / "examples/models" / config_name)
    features = FEATURES
    inputs = _inputs(2)
    tokenizer = None
    if config.type == "pepnet":
        from train.core.config import FlowConfig
        from train.core.dag import FeatureDag

        dag = FeatureDag(FlowConfig.from_yaml(str(REPO_ROOT / "examples/shared/feature_config_demo.yaml")))
        features = dag.feature_tuples()
        inputs = {name: torch.zeros(2, dtype=torch.long) for name, _, _ in features}
    if config.type in {"unimixer", "token_mixer_large", "rankmixer"}:
        tokenizer = FeatureTokenizer(
            features,
            token_dim=config.params["token_dim"],
            num_tokens=config.params["num_tokens"],
        )

    model = config.build(features, tokenizer=tokenizer)
    spec = get_output_spec(config.type, model, config.params)
    execution = model.forward_execution(inputs)

    assert spec["output_contract"].version == 1
    assert execution.outputs.names()
    assert set(execution.outputs.names()) == {
        output.name for output in spec["output_contract"].outputs
    }
    assert set(execution.nodes.names()) == set(spec["output_contract"].node_kinds)


def test_deepfm_forward():
    model = DeepFM(FEATURES, fm_k=8, deep_hidden_dims=[4])
    out = model(_inputs(3))
    assert out.tensor("pred").shape == (3, 1)


def test_feature_embeddings_first_pooling_accepts_sequences():
    embeddings = FeatureEmbeddings([("seq", 10, 4)], pooling_map={"seq": "first"})

    out = embeddings({"seq": torch.tensor([[1, 2], [3, 4]])})

    assert out.shape == (2, 4)


def test_gated_cross_network_forward():
    layer = GatedCrossNetwork(input_dim=8, num_layers=2)

    out = layer(torch.randn(3, 8))

    assert out.shape == (3, 8)


def test_mmoe_forward():
    model = MMoE(FEATURES, [8], 2, [8], 4, [("ctr", [4]), ("cvr", [4])])
    out = model(_inputs(3))
    assert out.tensor("ctr").shape == (3, 1)
    assert out.tensor("cvr").shape == (3, 1)


def test_rankup_forward_with_task_token_output_contract():
    from train.core.output_contract import parse_output_contract

    contract = parse_output_contract(
        {
            "version": 1,
            "graph": {
                "towers": [
                    {
                        "name": "ctr_logit",
                        "input": "task_0",
                        "kind": "binary_logit",
                        "hidden_dims": [4],
                    }
                ],
                "relations": [{"name": "ctr_prob", "op": "sigmoid", "inputs": ["ctr_logit"]}],
            },
            "objectives": [
                {
                    "name": "ctr_loss",
                    "source": "ctr_logit",
                    "label": "is_click",
                    "loss": {"type": "binary_cross_entropy_with_logits"},
                }
            ],
            "metrics": [{"name": "ctr_auc", "source": "ctr_logit", "label": "is_click", "type": "auc"}],
            "outputs": [{"name": "ctr", "source": "ctr_prob"}],
        }
    )
    model = RankUpModel(
        FEATURES,
        RankUpConfig(token_dim=4, num_sparse_tokens=2, num_blocks=1, num_task_tokens=1),
        task_config=None,
        output_contract=contract,
    )

    execution = model.forward_execution(_inputs(3))

    assert execution.nodes.tensor("ctr_logit").shape == (3, 1)
    assert execution.outputs.tensor("ctr").shape == (3, 1)


def test_hyformer_forward_with_output_contract():
    from train.core.output_contract import parse_output_contract
    from train.models.hyformer import HyFormerConfig, HyFormerModel

    contract = parse_output_contract(
        {
            "version": 1,
            "graph": {
                "towers": [
                    {
                        "name": "ctr_logit",
                        "kind": "binary_logit",
                        "hidden_dims": [4],
                    }
                ],
                "relations": [{"name": "ctr_prob", "op": "sigmoid", "inputs": ["ctr_logit"]}],
            },
            "objectives": [
                {
                    "name": "ctr_loss",
                    "source": "ctr_logit",
                    "label": "is_click",
                    "loss": {"type": "binary_cross_entropy_with_logits"},
                }
            ],
            "metrics": [
                {"name": "ctr_auc", "source": "ctr_logit", "label": "is_click", "type": "auc"}
            ],
            "outputs": [{"name": "ctr", "source": "ctr_prob"}],
        }
    )
    model = HyFormerModel(
        FEATURES,
        HyFormerConfig(d=4, d_ff=8, num_queries=2, num_layers=1),
        task_config=None,
        output_contract=contract,
    )

    execution = model.forward_execution(_inputs(3))

    assert execution.nodes.tensor("ctr_logit").shape == (3, 1)
    assert execution.outputs.tensor("ctr").shape == (3, 1)


def test_esmm_forward():
    model = ESMM(FEATURES, [8], [4], [4], [4], [4], [4])
    out = model(_inputs(3))
    assert out.tensor("click").shape == (3, 1)
    assert out.tensor("cvr").shape == (3, 1)
    assert out.tensor("ctcvr").shape == (3, 1)


def test_esmm_forward_with_configurable_tasks_and_relations():
    task_config = MultiTaskConfig(
        towers=[
            TowerConfig("view", [4], 1, Activation.RELU),
            TowerConfig("buy", [4], 1, Activation.RELU),
        ],
        relations=[TaskRelation("ctbuy", ["view", "buy"], "multiply")],
    )
    model = ESMM(FEATURES, [8], [], [], [], [], [], task_config=task_config)
    out = model(_inputs(3))

    assert model.task_names == ["view", "buy"]
    assert set(out.names()) == {"view", "buy", "ctbuy"}
    assert out.tensor("ctbuy").shape == (3, 1)


def test_esmm_output_spec_marks_relations_as_probabilities():
    spec = get_output_spec(
        "esmm",
        params={
            "task_config": {
                "towers": [
                    {"name": "view", "hidden_dims": [4]},
                    {"name": "buy", "hidden_dims": [4]},
                ],
                "relations": [{"target": "ctbuy", "sources": ["view", "buy"], "op": "multiply"}],
            }
        },
    )

    assert spec["output_kinds"] == {
        "view": "binary_logit",
        "buy": "binary_logit",
        "ctbuy": "probability",
    }


def test_esmm_relation_uses_probabilities_not_logits():
    relation = TaskRelation("ctbuy", ["view", "buy"], "multiply")
    outputs = ModelOutput.binary_logits(
        {
            "view": torch.tensor([[0.0], [2.0]]),
            "buy": torch.tensor([[0.0], [-2.0]]),
        }
    )

    derived = ESMM._apply_relation(relation, outputs)
    expected = torch.sigmoid(outputs.tensor("view")) * torch.sigmoid(outputs.tensor("buy"))

    assert torch.allclose(derived, expected)


def test_gdcn_esmm_relation_uses_probabilities_not_logits():
    relation = TaskRelation("ctbuy", ["view", "buy"], "multiply")
    outputs = ModelOutput.binary_logits(
        {
            "view": torch.tensor([[0.0], [2.0]]),
            "buy": torch.tensor([[0.0], [-2.0]]),
        }
    )

    derived = GDCNESMM._apply_relation(relation, outputs)
    expected = torch.sigmoid(outputs.tensor("view")) * torch.sigmoid(outputs.tensor("buy"))

    assert torch.allclose(derived, expected)


def test_gdcn_esmm_forward():
    model = GDCNESMM(
        FEATURES,
        cross_layers=2,
        deep_hidden_dims=[8],
        shared_bottom_dims=[8],
        click_hidden_dims=[4],
        cvr_hidden_dims=[4],
        detail_hidden_dims=[4],
        stock_hidden_dims=[4],
        stay_hidden_dims=[4],
    )

    out = model(_inputs(3))

    assert out.tensor("click").shape == (3, 1)
    assert out.tensor("cvr").shape == (3, 1)
    assert out.tensor("ctcvr").shape == (3, 1)


def test_gdcn_esmm_builds_from_model_config():
    config = ModelConfig.from_dict(
        {
            "type": "gdcn_esmm",
            "cross_layers": 2,
            "deep_hidden_dims": [8],
            "shared_bottom_dims": [8],
            "task_config": {
                "towers": [
                    {"name": "view", "hidden_dims": [4]},
                    {"name": "buy", "hidden_dims": [4]},
                ],
                "relations": [{"target": "ctbuy", "sources": ["view", "buy"], "op": "multiply"}],
            },
        }
    )

    model = config.build(FEATURES)
    out = model(_inputs(3))
    spec = get_output_spec(config.type, model, config.params)

    assert set(out.names()) == {"view", "buy", "ctbuy"}
    assert spec["task_names"] == ["view", "buy"]


def test_gate_nu_scales_to_paper_range():
    gate = GateNU(input_dim=3, hidden_dim=4, output_dim=2)

    out = gate(torch.randn(5, 3))

    assert out.shape == (5, 2)
    assert torch.all(out >= 0)
    assert torch.all(out <= 2)


def test_pepnet_forward_with_deep_without_shared_bottom():
    task_config = MultiTaskConfig(
        towers=[TowerConfig("click", [4], 1, Activation.RELU)],
        relations=[],
    )
    model = PEPNet(FEATURES, prior_dim=4, deep_hidden_dims=[8], task_config=task_config)

    out = model(_inputs(3))

    assert out.tensor("click").shape == (3, 1)


def test_unimixer_forward():
    from train.models.unimixer.model import UniMixerModel
    from train.models.unimixer.tokenizer import FeatureTokenizer

    token_dim = 4
    num_tokens = 2
    tokenizer = FeatureTokenizer(FEATURES, token_dim, num_tokens)
    task_config = MultiTaskConfig(
        towers=[
            TowerConfig("ctr", [8], 1, Activation.RELU),
            TowerConfig("cvr", [8], 1, Activation.RELU),
        ],
        relations=[],
    )
    model = UniMixerModel(
        tokenizer,
        token_dim,
        num_tokens,
        1,
        4,
        False,
        1.0,
        4,
        4,
        task_config,
        False,
    )
    out = model(_inputs(3))
    assert out.tensor("ctr").shape == (3, 1)
    assert out.tensor("cvr").shape == (3, 1)


def test_rankmixer_forward():
    from train.models.rankmixer.model import RankMixerModel
    from train.models.unimixer.tokenizer import FeatureTokenizer

    token_dim = 4
    num_tokens = 2
    tokenizer = FeatureTokenizer(FEATURES, token_dim, num_tokens)
    task_config = MultiTaskConfig(
        towers=[
            TowerConfig("ctr", [8], 1, Activation.RELU),
            TowerConfig("cvr", [8], 1, Activation.RELU),
        ],
        relations=[],
    )
    model = RankMixerModel(tokenizer, token_dim, num_tokens, 1, num_tokens, 1.0, task_config)

    out = model(_inputs(3))

    assert out.tensor("ctr").shape == (3, 1)
    assert out.tensor("cvr").shape == (3, 1)


def test_rankmixer_builds_from_model_config():
    from train.models.unimixer.tokenizer import FeatureTokenizer

    tokenizer = FeatureTokenizer(FEATURES, token_dim=4, num_tokens=2)
    config = ModelConfig.from_dict(
        {
            "type": "rankmixer",
            "token_dim": 4,
            "num_tokens": 2,
            "num_blocks": 1,
            "task_config": {"towers": [{"name": "ctr", "hidden_dims": [8]}]},
        }
    )

    model = config.build(FEATURES, tokenizer=tokenizer)
    out = model(_inputs(3))

    assert out.tensor("ctr").shape == (3, 1)


def test_rankmixer_rejects_non_residual_token_mixing_shape():
    from train.models.rankmixer.block import RankMixerBlock

    with pytest.raises(ValueError, match="num_heads == num_tokens"):
        RankMixerBlock(token_dim=4, num_tokens=2, num_heads=1, hidden_factor=1.0)


def test_unimixer_tokenizer_pools_first_and_flatten_sequences():
    from train.models.unimixer.tokenizer import FeatureTokenizer

    tokenizer = FeatureTokenizer(
        [("seq", 10, 4), ("flat", 10, 2)],
        token_dim=3,
        num_tokens=2,
        pooling_map={"seq": "first", "flat": "flatten"},
        seq_len_map={"flat": 2},
    )

    out = tokenizer(
        {
            "seq": torch.tensor([[1, 2], [3, 4]]),
            "flat": torch.tensor([[1, 2], [3, 4]]),
        }
    )

    assert out.shape == (2, 2, 3)


def test_unimixer_rejects_invalid_temperature():
    from train.models.unimixer.model import UniMixerModel
    from train.models.unimixer.tokenizer import FeatureTokenizer

    tokenizer = FeatureTokenizer(FEATURES, token_dim=4, num_tokens=2)
    task_config = MultiTaskConfig(
        towers=[TowerConfig("ctr", [8], 1, Activation.RELU)],
        relations=[],
    )
    model = UniMixerModel(
        tokenizer,
        4,
        2,
        1,
        4,
        False,
        1.0,
        4,
        4,
        task_config,
        False,
    )

    with pytest.raises(ValueError, match="temperature"):
        model(_inputs(3), temperature=0.0)


def test_unimixer_block_mode_is_constructor_state():
    from train.models.unimixer.block import UniMixerBlock

    block = UniMixerBlock(
        embed_dim=8,
        block_size=4,
        token_dim=4,
        num_tokens=2,
        use_lite=False,
        hidden_factor=1.0,
        num_basis=4,
        rank=4,
        use_siamese=True,
    )

    with pytest.raises(ValueError, match="siamese block requires"):
        block(torch.randn(2, 8), temperature=1.0)
