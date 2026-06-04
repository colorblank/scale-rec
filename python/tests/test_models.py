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

FEATURES = [("a", 10, 4), ("b", 5, 4)]


def _inputs(batch=3):
    return {
        "a": torch.tensor([1, 2, 3][:batch]),
        "b": torch.tensor([0, 1, 2][:batch]),
    }


def test_lr_forward():
    model = LogisticRegression(FEATURES)
    out = model(_inputs(3))
    assert out["pred"].shape == (3, 1)


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
    assert spec["tasks"][0].weight == 0.5


def test_deepfm_forward():
    model = DeepFM(FEATURES, fm_k=8, deep_hidden_dims=[4])
    out = model(_inputs(3))
    assert out["pred"].shape == (3, 1)


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
    assert out["ctr"].shape == (3, 1)
    assert out["cvr"].shape == (3, 1)


def test_esmm_forward():
    model = ESMM(FEATURES, [8], [4], [4], [4], [4], [4])
    out = model(_inputs(3))
    assert out["click"].shape == (3, 1)
    assert out["cvr"].shape == (3, 1)
    assert out["ctcvr"].shape == (3, 1)


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
    assert set(out) == {"view", "buy", "ctbuy"}
    assert out["ctbuy"].shape == (3, 1)


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

    assert out["click"].shape == (3, 1)
    assert out["cvr"].shape == (3, 1)
    assert out["ctcvr"].shape == (3, 1)


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

    assert set(out) == {"view", "buy", "ctbuy"}
    assert spec["task_names"] == ["view", "buy"]


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
    assert out["ctr"].shape == (3, 1)
    assert out["cvr"].shape == (3, 1)


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
