use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::feats::config::{FlowConfig, PoolingStrategy};
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
use scale_rec::feats::ops::Fv;
use scale_rec::feats::schema::FeatureDType;
use scale_rec::layers::embedding::FeatureSpec;
use scale_rec::models::{lr, Model};

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
    assert_eq!(outputs["pred"].shape().dims(), &[2, 1]);
}
