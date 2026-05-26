import pytest

from train.core.config import FlowConfig
from train.core.dag import FeatureDag


def test_dag_from_yaml():
    config = FlowConfig.from_yaml("examples/feature_config.yaml")
    dag = FeatureDag(config)
    features = dag.feature_tuples()
    assert len(features) == 5
    names = [f[0] for f in features]
    assert "user_id_idx" in names
    assert "user_age_bucket" in names
    assert "item_category_idx" in names
    assert "user_tag_mapped" in names
    assert "user_category_cross" in names
    assert dag.feature_schemas["user_id_idx"].dtype.tag == "int"
    assert dag.feature_schemas["user_tag_mapped"].dtype.tag == "list"
    assert dag.feature_schemas["user_tag_mapped"].cardinality == 6
    assert dag.validation_report.source_count == 5
    assert any(issue.code == "orphan_output" for issue in dag.validation_report.warnings)


def test_dag_execute():
    config = FlowConfig.from_yaml("examples/feature_config.yaml")
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
    config = FlowConfig.from_yaml("examples/feature_config.yaml")
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
    assert "user_id_idx" in tensors
    assert tensors["user_id_idx"].shape == (2,)
    assert str(tensors["user_id_idx"].dtype) == "torch.int64"


def test_dag_rejects_non_integer_embeddable_feature():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "category",
                "dtype": "string",
                "default_val": "unknown",
            }
        ],
        "operators": [
            {
                "name": "concat",
                "op_type": "StringConcat",
                "inputs": ["category"],
                "outputs": ["category_text"],
                "params": {"separator": "_"},
                "embed": {"vocab_size": 10, "embed_dim": 4},
            }
        ],
    }
    with pytest.raises(ValueError, match=r"must be int or list\[int\]"):
        FeatureDag(FlowConfig.from_dict(raw))


def test_strict_validation_rejects_warning_level_issues():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "unused",
                "dtype": "string",
                "default_val": "",
            }
        ],
        "operators": [],
    }
    with pytest.raises(ValueError, match="strict validation failed"):
        FeatureDag(FlowConfig.from_dict(raw), strict_validation=True)
