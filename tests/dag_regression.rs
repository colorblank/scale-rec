use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::feats::config::{FlowConfig, PoolingStrategy};
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
use scale_rec::feats::ops::Fv;
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
