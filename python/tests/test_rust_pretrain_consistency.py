"""一致性测试：Rust FeatSession 预处理结果与 Python FeatureDag 对比。"""

import pytest
import torch

from train.core.config import FlowConfig
from train.core.dag import FeatureDag

FIXTURE_DIR = __file__.rsplit("/", 1)[0] if "/" in __file__ else __file__.rsplit("\\", 1)[0]
CONFIG_PATH = f"{FIXTURE_DIR}/../../examples/shared/feature_config_demo.yaml"


def _make_test_data() -> dict[str, list[str]]:
    return {
        "user_id": ["123", "456", "789"],
        "item_id": ["101", "202", "303"],
        "scene": ["1", "2", "1"],
        "rec_algo": ["algo_a", "algo_b", "algo_a"],
        "is_new_user": ["yes", "no", "yes"],
        "asset_level": ["rich", "poor", "rich"],
        "city": ["bj", "sh", "gz"],
        "investment_horizon": ["short", "long", "medium"],
        "stay_time": ["120", "600", "300"],
    }


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("feat_engine"),
    reason="feat_engine not built (run 'maturin develop' in python/rust_feat_bridge)",
)
def test_rust_pretrain_consistency():
    fc = FlowConfig.from_yaml(CONFIG_PATH)
    dag_rust = FeatureDag(fc, use_rust=True, config_path=CONFIG_PATH)
    dag_py = FeatureDag(fc)
    test_data = _make_test_data()
    result_rust = dag_rust.preprocess_batch(test_data)
    result_py = dag_py.preprocess_batch(test_data)
    assert set(result_rust.keys()) == set(result_py.keys())
    for key in result_rust:
        assert torch.equal(result_rust[key], result_py[key]), (
            f"feature '{key}' mismatch: rust={result_rust[key].tolist()}, "
            f"py={result_py[key].tolist()}"
        )
