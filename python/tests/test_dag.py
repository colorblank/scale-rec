from train.config import FlowConfig
from train.dag import FeatureDag


def test_dag_from_yaml():
    config = FlowConfig.from_yaml("../examples/feature_config.yaml")
    dag = FeatureDag(config)
    features = dag.feature_tuples()
    assert len(features) == 5
    names = [f[0] for f in features]
    assert "user_id" in names
    assert "user_age_bucket" in names
    assert "item_category_idx" in names
    assert "user_tag_mapped" in names
    assert "user_category_cross" in names


def test_dag_execute():
    config = FlowConfig.from_yaml("../examples/feature_config.yaml")
    dag = FeatureDag(config)
    raw = {
        "user_id": 42,
        "user_age": 28.5,
        "item_category": "electronics",
        "user_tags": "sports#1|gaming#0.8",
        "item_price": 5999.0,
    }
    result = dag.execute(raw)
    assert result.features["user_age_bucket"] == 2
    assert result.features["item_category_idx"] == 1


def test_preprocess_batch():
    config = FlowConfig.from_yaml("../examples/feature_config.yaml")
    dag = FeatureDag(config)
    rows = [
        {
            "user_id": 42,
            "user_age": 28.5,
            "item_category": "electronics",
            "user_tags": "sports#1",
            "item_price": 5999.0,
        },
        {
            "user_id": 7,
            "user_age": 20.0,
            "item_category": "books",
            "user_tags": "music#1",
            "item_price": 100.0,
        },
    ]
    tensors = dag.preprocess_batch(rows)
    assert "user_id" in tensors
    assert tensors["user_id"].shape == (2,)
    assert str(tensors["user_id"].dtype) == "torch.int64"
