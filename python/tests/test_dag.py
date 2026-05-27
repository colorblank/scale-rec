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


def test_preprocess_batch_uses_configured_flatten_seq_len():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "seq",
                "dtype": {"list": {"dtype": "int", "length": 4}},
                "default_val": "0",
            }
        ],
        "operators": [
            {
                "name": "seq_hash",
                "op_type": "FeatureHash",
                "inputs": ["seq"],
                "outputs": ["seq_idx"],
                "params": {"vocab_size": 32, "num_hashes": 1},
                "embed": {"vocab_size": 32, "embed_dim": 4, "pooling": "flatten", "seq_len": 4},
            }
        ],
    }
    dag = FeatureDag(FlowConfig.from_dict(raw))
    tensors = dag.preprocess_batch(
        [
            {"seq": [1, 2]},
            {"seq": [3, 4, 5, 6, 7]},
        ]
    )

    assert tensors["seq_idx"].shape == (2, 4)


def test_preprocess_batch_list_valued_fallback_correctness():
    raw = {
        "version": "1.0.0",
        "sources": [
            {"name": "u_tags", "dtype": {"list": {"dtype": "string", "length": 2}}, "default_val": "a"},
            {"name": "i_tags", "dtype": {"list": {"dtype": "string", "length": 2}}, "default_val": "b"},
        ],
        "operators": [
            {
                "name": "cross",
                "op_type": "CrossFeature",
                "inputs": ["u_tags", "i_tags"],
                "outputs": ["tag_cross"],
                "params": {"cross_type": "cartesian"},
            },
            {
                "name": "hash",
                "op_type": "FeatureHash",
                "inputs": ["tag_cross"],
                "outputs": ["tag_cross_idx"],
                "params": {"vocab_size": 16},
                "embed": {"vocab_size": 16, "embed_dim": 4, "pooling": "mean"},
            }
        ],
    }
    dag = FeatureDag(FlowConfig.from_dict(raw))
    rows = [
        {"u_tags": ["u1", "u2"], "i_tags": ["i1"]},
        {"u_tags": ["u3"], "i_tags": ["i2", "i3"]},
    ]
    tensors = dag.preprocess_batch(rows)
    assert tensors["tag_cross_idx"].shape == (2, 4)
    # Check that it matches execute for each row
    for i, r in enumerate(rows):
        res = dag.execute(r)
        expected = res.features["tag_cross_idx"]
        assert tensors["tag_cross_idx"][i].tolist()[:len(expected)] == expected


def test_enum_source_can_be_mapped_to_embedding_index():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "category",
                "dtype": {
                    "enum": {
                        "values": ["unknown", "books", "fashion"],
                        "default": "unknown",
                        "oov": "unknown",
                    }
                },
                "default_val": "unknown",
            }
        ],
        "operators": [
            {
                "name": "category_map",
                "op_type": "DictMapper",
                "inputs": ["category"],
                "outputs": ["category_idx"],
                "params": {"mapping": {"books": 1, "fashion": 2}, "default_idx": 0},
                "embed": {"vocab_size": 3, "embed_dim": 4},
            }
        ],
    }
    dag = FeatureDag(FlowConfig.from_dict(raw))

    assert dag.feature_schemas["category"].dtype.tag == "enum"
    assert dag.feature_schemas["category_idx"].dimension == 1
    assert dag.preprocess_batch([{"category": "books"}, {"category": "unknown"}])[
        "category_idx"
    ].tolist() == [1, 0]


def test_list_embedding_uses_schema_fixed_dimension_for_mean_pooling():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "tags",
                "dtype": {"list": {"item_dtype": "string", "max_len": 3}},
                "default_val": "unknown",
            }
        ],
        "operators": [
            {
                "name": "hash",
                "op_type": "FeatureHash",
                "inputs": ["tags"],
                "outputs": ["tag_idx"],
                "params": {"vocab_size": 16},
                "embed": {"vocab_size": 16, "embed_dim": 4, "pooling": "mean"},
            }
        ],
    }
    dag = FeatureDag(FlowConfig.from_dict(raw))
    tensors = dag.preprocess_batch(
        [
            {"tags": ["a"]},
            {"tags": ["b", "c", "d", "e"]},
        ]
    )

    assert dag.feature_schemas["tag_idx"].dimension == 3
    assert dag.feature_seq_lens()["tag_idx"] == 3
    assert tensors["tag_idx"].shape == (2, 3)


def test_feature_hash_uses_all_inputs_for_list_dimension():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "tags",
                "dtype": {"list": {"dtype": "string", "length": 2}},
                "default_val": "unknown",
            },
            {
                "name": "category",
                "dtype": "string",
                "default_val": "unknown",
            },
            {
                "name": "user_id",
                "dtype": "int",
                "default_val": "0",
            },
        ],
        "operators": [
            {
                "name": "hash",
                "op_type": "FeatureHash",
                "inputs": ["tags", "category", "user_id"],
                "outputs": ["mixed_idx"],
                "params": {"vocab_size": 16},
                "embed": {"vocab_size": 16, "embed_dim": 4, "pooling": "mean"},
            }
        ],
    }
    dag = FeatureDag(FlowConfig.from_dict(raw))

    assert dag.feature_schemas["mixed_idx"].dimension == 4
    tensors = dag.preprocess_batch(
        [
            {"tags": ["a", "b"], "category": "books", "user_id": 1},
            {"tags": ["c", "d"], "category": "fashion", "user_id": 2},
        ]
    )
    assert tensors["mixed_idx"].shape == (2, 4)


def test_cross_feature_rejects_more_than_two_inputs():
    raw = {
        "version": "1.0.0",
        "sources": [
            {"name": "a", "dtype": {"list": {"dtype": "int", "length": 2}}, "default_val": "0"},
            {"name": "b", "dtype": {"list": {"dtype": "int", "length": 2}}, "default_val": "0"},
            {"name": "c", "dtype": {"list": {"dtype": "int", "length": 2}}, "default_val": "0"},
        ],
        "operators": [
            {
                "name": "cross",
                "op_type": "CrossFeature",
                "inputs": ["a", "b", "c"],
                "outputs": ["abc_cross"],
                "params": {"cross_type": "cartesian"},
            }
        ],
    }

    with pytest.raises(ValueError, match="exactly 2 inputs"):
        FeatureDag(FlowConfig.from_dict(raw))


def test_list_embedding_respects_truncation_side():
    raw_head = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "tags",
                "dtype": {"list": {"item_dtype": "string", "max_len": 3}},
                "default_val": "unknown",
            }
        ],
        "operators": [
            {
                "name": "hash",
                "op_type": "FeatureHash",
                "inputs": ["tags"],
                "outputs": ["tag_idx"],
                "params": {"vocab_size": 16},
                "embed": {"vocab_size": 16, "embed_dim": 4, "pooling": "mean", "truncation": "head"},
            }
        ],
    }
    dag_head = FeatureDag(FlowConfig.from_dict(raw_head))
    tensors_head = dag_head.preprocess_batch([{"tags": ["b", "c", "d", "e"]}])

    raw_tail = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "tags",
                "dtype": {"list": {"item_dtype": "string", "max_len": 3}},
                "default_val": "unknown",
            }
        ],
        "operators": [
            {
                "name": "hash",
                "op_type": "FeatureHash",
                "inputs": ["tags"],
                "outputs": ["tag_idx"],
                "params": {"vocab_size": 16},
                "embed": {"vocab_size": 16, "embed_dim": 4, "pooling": "mean", "truncation": "tail"},
            }
        ],
    }
    dag_tail = FeatureDag(FlowConfig.from_dict(raw_tail))
    tensors_tail = dag_tail.preprocess_batch([{"tags": ["b", "c", "d", "e"]}])

    head_vals = tensors_head["tag_idx"][0].tolist()
    tail_vals = tensors_tail["tag_idx"][0].tolist()
    assert head_vals != tail_vals

    res_b = dag_head.execute({"tags": ["b"]}).features["tag_idx"]
    res_c = dag_head.execute({"tags": ["c"]}).features["tag_idx"]
    res_d = dag_head.execute({"tags": ["d"]}).features["tag_idx"]
    res_e = dag_head.execute({"tags": ["e"]}).features["tag_idx"]

    assert head_vals == [res_b[0], res_c[0], res_d[0]]
    assert tail_vals == [res_c[0], res_d[0], res_e[0]]


def test_dag_rejects_fractional_int_default():
    raw = {
        "version": "1.0.0",
        "sources": [
            {
                "name": "user_id",
                "dtype": "int",
                "default_val": "12.9",
            }
        ],
        "operators": [],
    }

    with pytest.raises(ValueError, match="does not match dtype"):
        FeatureDag(FlowConfig.from_dict(raw))


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
