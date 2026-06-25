use std::collections::{HashMap, HashSet};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::feats::builder::DagBuilder;
use scale_rec::feats::config::{FlowConfig, PoolingStrategy, Role, SourceKind};
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
use scale_rec::feats::executor::DagExecutor;
use scale_rec::feats::ops::Fv;
use scale_rec::feats::schema::FeatureDType;
use scale_rec::layers::embedding::FeatureSpec;
use scale_rec::models::{lr, Model};

#[test]
fn dag_rejects_unknown_operator_param() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: user_id
    dtype: int
    default_val: "0"
operators:
  - name: user_hash
    op_type: FeatureHash
    inputs: [user_id]
    outputs: [user_id_idx]
    params:
      vocab_size: 100
      typo: 1
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let err = match FeatureDag::from_config(config, false, None) {
        Ok(_) => panic!("unknown operator param should fail"),
        Err(err) => err,
    };
    assert!(err.contains("unknown field 'typo'"));
}

#[test]
fn dag_rejects_missing_required_operator_param() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: user_id
    dtype: int
    default_val: "0"
operators:
  - name: user_hash
    op_type: FeatureHash
    inputs: [user_id]
    outputs: [user_id_idx]
    params: {}
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let err = match FeatureDag::from_config(config, false, None) {
        Ok(_) => panic!("missing operator param should fail"),
        Err(err) => err,
    };
    assert!(err.contains("missing required field 'vocab_size'"));
}

#[test]
fn flow_config_normalizes_label_sources_to_label_role() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: user_id
    dtype: string
    default_val: ""
  - name: is_click
    source: Label
    dtype: int
    default_val: "0"
operators: []
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    assert!(matches!(config.sources[1].source, Some(SourceKind::Label)));
    assert_eq!(config.sources[1].role, Role::Label);

    let artifact = DagBuilder::build(config).unwrap();
    assert!(artifact.sources.contains_key("user_id"));
    assert!(!artifact.sources.contains_key("is_click"));
}

#[test]
fn dag_execute_uses_compiled_plan_ops() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: user_id
    dtype: int
    default_val: "7"
operators:
  - name: user_hash
    op_type: FeatureHash
    inputs: [user_id]
    outputs: [user_id_idx]
    params:
      vocab_size: 100
      num_hashes: 1
    embed:
      vocab_size: 100
      embed_dim: 4
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();
    let out = dag
        .execute(&HashMap::<String, FeatureValue>::new())
        .unwrap();
    assert!(matches!(out.features.get("user_id_idx"), Some(Fv::Int(_))));
    assert!(matches!(
        dag.feature_schemas.get("user_id_idx").unwrap().dtype,
        FeatureDType::Int
    ));
    assert_eq!(dag.validation_report.errors().count(), 0);
}

#[test]
fn dag_executor_reports_dict_mapper_default_hits_by_configured_operator_name() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: scene
    dtype: string
    default_val: unknown
operators:
  - name: scene_mapper
    op_type: DictMapper
    inputs: [scene]
    outputs: [scene_idx]
    params:
      mapping:
        home: 1
        same_as_default: 99
      default_idx: 99
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let artifact = DagBuilder::build(config).unwrap();
    let executor = DagExecutor::new(
        artifact.plan,
        artifact.sources,
        artifact.execution_order,
        artifact.data_sources,
    );
    let columns = HashMap::from([(
        "scene".to_string(),
        vec![
            Fv::Str("home".into()),
            Fv::Str("missing".into()),
            Fv::Str("same_as_default".into()),
        ],
    )]);

    let (_, stats) = executor
        .execute_plan_with_stats(&columns, &HashSet::new(), &HashMap::new())
        .unwrap();

    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].operator, "scene_mapper");
    assert_eq!(stats[0].stats.dict_mapper_default_hits, 1);
}

#[test]
fn dag_executes_log1p_operator() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: raw_score
    dtype: float
    default_val: "0"
operators:
  - name: score_log1p
    op_type: Log1p
    inputs: [raw_score]
    outputs: [score_log]
    params: {}
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();
    let mut raw = HashMap::new();
    raw.insert("raw_score".to_string(), FeatureValue::Float(5999.0));
    let out = dag.execute(&raw).unwrap();

    match out.features.get("score_log") {
        Some(Fv::Float(value)) => assert!((*value - 8.699_515).abs() < 1e-6),
        other => panic!("expected log1p float output, got {:?}", other),
    }
    assert_eq!(
        dag.feature_schemas.get("score_log").unwrap().dtype,
        FeatureDType::Float
    );
}

#[test]
fn dag_rejects_non_integer_embeddable_feature() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: category
    dtype: string
    default_val: unknown
operators:
  - name: concat
    op_type: StringConcat
    inputs: [category]
    outputs: [category_text]
    params:
      separator: "_"
    embed:
      vocab_size: 10
      embed_dim: 4
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let err = match FeatureDag::from_config(config, false, None) {
        Ok(_) => panic!("expected non-integer embeddable feature to be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("must be int or list"));
}

#[test]
fn dag_records_warning_level_validation_issues() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: unused
    dtype: string
    default_val: ""
operators: []
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();
    let warnings: Vec<_> = dag.validation_report.warnings().collect();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].code, "orphan_source");
}

#[test]
fn dag_rejects_fractional_int_default_values() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: user_id
    dtype: int
    default_val: "12.9"
operators: []
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let err = match FeatureDag::from_config(config, false, None) {
        Ok(_) => panic!("expected fractional int default to be rejected"),
        Err(err) => err,
    };
    assert!(err.contains("does not match dtype"));
}

#[test]
fn rust_embedding_accepts_sequence_pooling_inputs() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut spec = FeatureSpec::new("seq_ids".into(), 32, 4);
    spec.pooling = PoolingStrategy::Mean;
    let model = lr::LogisticRegression::new(vb, &[spec]).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(
        "seq_ids".into(),
        Tensor::from_slice(&[1u32, 2, 3, 4, 5, 6], (2, 3), &device).unwrap(),
    );
    let outputs = model.forward(&inputs).unwrap();
    assert_eq!(outputs.tensor("pred").unwrap().shape().dims(), &[2, 1]);
}

#[test]
fn dag_tracks_enum_and_fixed_list_dimensions() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: category
    dtype:
      enum:
        values: [unknown, books, fashion]
        default: unknown
        oov: unknown
    default_val: unknown
  - name: tags
    dtype:
      list:
        item_dtype: string
        max_len: 3
    default_val: unknown
operators:
  - name: category_map
    op_type: DictMapper
    inputs: [category]
    outputs: [category_idx]
    params:
      default_idx: 0
      mapping:
        books: 1
        fashion: 2
    embed:
      vocab_size: 3
      embed_dim: 4
  - name: tag_hash
    op_type: FeatureHash
    inputs: [tags]
    outputs: [tag_idx]
    params:
      vocab_size: 16
    embed:
      vocab_size: 16
      embed_dim: 4
      pooling: mean
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();

    assert!(matches!(
        dag.feature_schemas.get("category").unwrap().dtype,
        FeatureDType::Enum { .. }
    ));
    assert_eq!(dag.feature_schemas.get("tag_idx").unwrap().dimension, 3);
}

#[test]
fn dag_rejects_cross_feature_with_too_many_inputs() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: a
    dtype:
      list:
        item_dtype: int
        max_len: 2
    default_val: "0"
  - name: b
    dtype:
      list:
        item_dtype: int
        max_len: 2
    default_val: "0"
  - name: c
    dtype:
      list:
        item_dtype: int
        max_len: 2
    default_val: "0"
operators:
  - name: cross
    op_type: CrossFeature
    inputs: [a, b, c]
    outputs: [abc_cross]
    params:
      cross_type: cartesian
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let err = match FeatureDag::from_config(config, false, None) {
        Ok(_) => panic!("expected CrossFeature arity validation to fail"),
        Err(err) => err,
    };
    assert!(err.contains("exactly 2 inputs"));
}

#[test]
fn dag_tracks_feature_hash_dimension_across_all_inputs() {
    let yaml = r#"
version: 1.0.0
sources:
  - name: tags
    dtype:
      list:
        item_dtype: string
        max_len: 2
    default_val: unknown
  - name: category
    dtype: string
    default_val: unknown
  - name: user_id
    dtype: int
    default_val: "0"
operators:
  - name: hash
    op_type: FeatureHash
    inputs: [tags, category, user_id]
    outputs: [mixed_idx]
    params:
      vocab_size: 16
    embed:
      vocab_size: 16
      embed_dim: 4
      pooling: mean
"#;
    let config = FlowConfig::from_yaml(yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();
    assert_eq!(dag.feature_schemas.get("mixed_idx").unwrap().dimension, 4);
}
