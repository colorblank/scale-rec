import torch

from train.models.deepfm import DeepFM
from train.models.esmm import ESMM
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


def test_deepfm_forward():
    model = DeepFM(FEATURES, fm_k=8, deep_hidden_dims=[4])
    out = model(_inputs(3))
    assert out["pred"].shape == (3, 1)


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


def test_unimixer_forward():
    from train.layers.towers import Activation, MultiTaskConfig, TowerConfig
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
