from pathlib import Path

import pytest

from train.core.config import FlowConfig, ModelConfig, SourceKind
from train.core.dag import FeatureDag

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def test_operator_params_reject_unknown_field():
    raw = {
        "version": "1.0.0",
        "sources": [{"name": "user_id", "dtype": "string", "default_val": ""}],
        "operators": [
            {
                "name": "user_hash",
                "op_type": "FeatureHash",
                "inputs": ["user_id"],
                "outputs": ["user_id_idx"],
                "params": {"vocab_size": 10, "typo": 1},
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown field"):
        FlowConfig.from_dict(raw)


def test_label_source_is_normalized_to_label_role():
    raw = {
        "version": "1.0.0",
        "sources": [
            {"name": "user_id", "dtype": "string", "default_val": ""},
            {
                "name": "is_click",
                "source": "Label",
                "dtype": "int",
                "default_val": "0",
            },
        ],
        "operators": [],
    }
    config = FlowConfig.from_dict(raw)

    assert config.sources[1].source is SourceKind.LABEL
    assert [s.name for s in config.label_sources] == ["is_click"]
    assert [s.name for s in config.feature_sources] == ["user_id"]


def test_operator_params_reject_missing_required_field():
    raw = {
        "version": "1.0.0",
        "sources": [{"name": "user_id", "dtype": "string", "default_val": ""}],
        "operators": [
            {
                "name": "user_hash",
                "op_type": "FeatureHash",
                "inputs": ["user_id"],
                "outputs": ["user_id_idx"],
                "params": {},
            }
        ],
    }
    with pytest.raises(ValueError, match="missing required"):
        FlowConfig.from_dict(raw)


def test_model_params_reject_wrong_type():
    with pytest.raises(TypeError, match="cross_layers must be int"):
        ModelConfig.from_dict({"type": "gdcn_esmm", "cross_layers": "3"})


def test_dag_from_yaml():
    config = FlowConfig.from_yaml(str(FIXTURE_DIR / "golden_feature_config.yaml"))
    dag = FeatureDag(config)
    features = dag.feature_tuples()
    assert len(features) == 5
    names = [f[0] for f in features]
    assert "age_bucket" in names
    assert "category_idx" in names
    assert "user_tag_idx" in names
    assert "tag_cross_idx" in names
    assert "session_idx" in names
    assert dag.feature_schemas["category_idx"].dtype.tag == "int"
    assert dag.feature_schemas["user_tag_idx"].dtype.tag == "list"
    assert dag.feature_schemas["user_tag_idx"].cardinality == 4
    assert dag.validation_report.source_count == 12
    assert any(issue.code == "orphan_output" for issue in dag.validation_report.warnings)


def test_dag_execute():
    config = FlowConfig.from_yaml(str(FIXTURE_DIR / "golden_feature_config.yaml"))
    dag = FeatureDag(config)
    raw = {
        "user_age": 28.5,
        "item_category": "electronics",
        "user_tags": "sports#1|gaming#0.8",
        "item_tags": "sports#1",
        "session_id": "abc",
        "json_tags_src": '[{"tag":"x"}]',
        "split_src": "a|b",
        "expr_v0": 1.5,
        "expr_v1": 2.0,
        "seq_src": [1, 2, 3],
        "concat_src1": "foo",
        "concat_src2": "bar",
    }
    result = dag.execute(raw)
    assert result.features["age_bucket"] == 2
    assert result.features["category_idx"] == 1


def test_dag_executes_log1p_operator():
    raw = {
        "version": "1.0.0",
        "sources": [{"name": "raw_score", "dtype": "float", "default_val": "0"}],
        "operators": [
            {
                "name": "score_log1p",
                "op_type": "Log1p",
                "inputs": ["raw_score"],
                "outputs": ["score_log"],
                "params": {},
            }
        ],
    }
    dag = FeatureDag(FlowConfig.from_dict(raw))
    result = dag.execute({"raw_score": 5999.0})

    assert abs(result.features["score_log"] - 8.699515) < 1e-6
    assert dag.feature_schemas["score_log"].dtype.tag == "float"


def test_preprocess_batch():
    config = FlowConfig.from_yaml(str(FIXTURE_DIR / "golden_feature_config.yaml"))
    dag = FeatureDag(config)
    rows = [
        {
            "user_age": 28.5,
            "item_category": "electronics",
            "user_tags": "sports#1",
            "item_tags": "sports#1",
            "session_id": "abc",
            "json_tags_src": '[{"tag":"x"}]',
            "split_src": "a|b",
            "expr_v0": 1.5,
            "expr_v1": 2.0,
            "seq_src": [1, 2, 3],
            "concat_src1": "foo",
            "concat_src2": "bar",
        },
        {
            "user_age": 20.0,
            "item_category": "books",
            "user_tags": "music#1",
            "item_tags": "music#1",
            "session_id": "def",
            "json_tags_src": '[{"tag":"y"}]',
            "split_src": "c|d",
            "expr_v0": 2.0,
            "expr_v1": 3.0,
            "seq_src": [4, 5, 6],
            "concat_src1": "baz",
            "concat_src2": "qux",
        },
    ]
    tensors = dag.preprocess_batch(rows)
    assert "category_idx" in tensors
    assert tensors["category_idx"].shape == (2,)
    assert str(tensors["category_idx"].dtype) == "torch.int64"


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
            {
                "name": "u_tags",
                "dtype": {"list": {"dtype": "string", "length": 2}},
                "default_val": "a",
            },
            {
                "name": "i_tags",
                "dtype": {"list": {"dtype": "string", "length": 2}},
                "default_val": "b",
            },
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
            },
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
        assert tensors["tag_cross_idx"][i].tolist()[: len(expected)] == expected


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
                "embed": {
                    "vocab_size": 16,
                    "embed_dim": 4,
                    "pooling": "mean",
                    "truncation": "head",
                },
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
                "embed": {
                    "vocab_size": 16,
                    "embed_dim": 4,
                    "pooling": "mean",
                    "truncation": "tail",
                },
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
