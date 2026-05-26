use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

use scale_rec::layers::embedding::FeatureSpec;
use scale_rec::layers::towers::{
    Activation, MultiTaskConfig, RelationOp, TaskRelation, TowerConfig,
};
use scale_rec::models::{deepfm, esmm, gdcn_esmm, lr, mmoe, Model, ModelConfig};

fn dummy_features() -> Vec<FeatureSpec> {
    vec![
        FeatureSpec::new("a".into(), 10, 4),
        FeatureSpec::new("b".into(), 5, 4),
    ]
}

fn dummy_inputs(batch: usize) -> HashMap<String, Tensor> {
    let mut m = HashMap::new();
    m.insert(
        "a".into(),
        Tensor::from_slice(&[1u32, 2, 3][..batch], batch, &Device::Cpu).unwrap(),
    );
    m.insert(
        "b".into(),
        Tensor::from_slice(&[0u32, 1, 2][..batch], batch, &Device::Cpu).unwrap(),
    );
    m
}

fn vb() -> VarBuilder<'static> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    // leak to get 'static
    Box::leak(Box::new(varmap));
    vb
}

#[test]
fn test_lr_forward_shape() {
    let model = lr::LogisticRegression::new(vb(), &dummy_features()).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    let pred = out.get("pred").unwrap();
    assert_eq!(pred.dims(), &[3, 1]);
}

#[test]
fn test_deepfm_forward_shape() {
    let model = deepfm::DeepFM::new(vb(), &dummy_features(), 8, &[4]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out["pred"].dims(), &[3, 1]);
}

#[test]
fn test_mmoe_forward_shape() {
    let task_cfgs = vec![("ctr".into(), vec![4]), ("cvr".into(), vec![4])];
    let model = mmoe::MMoE::new(vb(), &dummy_features(), &[8], 2, &[8], 4, &task_cfgs).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out["ctr"].dims(), &[3, 1]);
    assert_eq!(out["cvr"].dims(), &[3, 1]);
}

#[test]
fn test_esmm_forward_shape() {
    let model =
        esmm::ESMM::new(vb(), &dummy_features(), &[8], &[4], &[4], &[4], &[4], &[4]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 9);
    assert_eq!(out["click"].dims(), &[3, 1]);
    assert_eq!(out["cvr"].dims(), &[3, 1]);
    assert_eq!(out["ctcvr"].dims(), &[3, 1]);
}

#[test]
fn test_esmm_forward_with_configurable_tasks_and_relations() {
    let task_config = MultiTaskConfig {
        towers: vec![
            TowerConfig {
                name: "view".into(),
                hidden_dims: vec![4],
                output_dim: 1,
                activation: Activation::Relu,
            },
            TowerConfig {
                name: "buy".into(),
                hidden_dims: vec![4],
                output_dim: 1,
                activation: Activation::Relu,
            },
        ],
        relations: vec![TaskRelation {
            target: "ctbuy".into(),
            sources: vec!["view".into(), "buy".into()],
            op: RelationOp::Multiply,
        }],
    };
    let model = esmm::ESMM::with_task_config(vb(), &dummy_features(), &[8], &task_config).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out["view"].dims(), &[3, 1]);
    assert_eq!(out["buy"].dims(), &[3, 1]);
    assert_eq!(out["ctbuy"].dims(), &[3, 1]);
}

#[test]
fn test_gdcn_esmm_forward_shape() {
    let model = gdcn_esmm::GDCNESMM::new(
        vb(),
        &dummy_features(),
        2,
        &[8],
        &[8],
        &[4],
        &[4],
        &[4],
        &[4],
        &[4],
    )
    .unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 9);
    assert_eq!(out["click"].dims(), &[3, 1]);
    assert_eq!(out["cvr"].dims(), &[3, 1]);
    assert_eq!(out["ctcvr"].dims(), &[3, 1]);
}

#[test]
fn test_modelconfig_build_gdcn_esmm() {
    let params = serde_yaml::from_str(
        r#"
cross_layers: 2
deep_hidden_dims: [8]
shared_bottom_dims: [8]
task_config:
  towers:
    - {name: view, hidden_dims: [4], output_dim: 1, activation: relu}
    - {name: buy, hidden_dims: [4], output_dim: 1, activation: relu}
  relations:
    - {target: ctbuy, sources: [view, buy], op: multiply}
"#,
    )
    .unwrap();
    let cfg = ModelConfig {
        model_type: "gdcn_esmm".into(),
        params,
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out["view"].dims(), &[2, 1]);
    assert_eq!(out["buy"].dims(), &[2, 1]);
    assert_eq!(out["ctbuy"].dims(), &[2, 1]);
}

#[test]
fn test_modelconfig_build_lr() {
    let cfg = ModelConfig {
        model_type: "lr".into(),
        params: serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("pred"));
}

#[test]
fn test_modelconfig_build_deepfm() {
    let mut params = serde_yaml::Mapping::new();
    params.insert("fm_k".into(), serde_yaml::Value::Number(8.into()));
    let mut dims = serde_yaml::Sequence::new();
    dims.push(serde_yaml::Value::Number(4.into()));
    params.insert("deep_hidden_dims".into(), serde_yaml::Value::Sequence(dims));
    let cfg = ModelConfig {
        model_type: "deepfm".into(),
        params: serde_yaml::Value::Mapping(params),
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("pred"));
}

#[test]
fn test_unimixer_forward_shape() {
    use scale_rec::models::unimixer::{model::UniMixerModel, tokenizer::FeatureTokenizer};

    let features = dummy_features();
    let token_dim = 4;
    let num_tokens = 2;
    let vb = vb();
    let tokenizer =
        FeatureTokenizer::new(vb.pp("tokenizer"), &features, token_dim, num_tokens).unwrap();
    let task_config = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "ctr".into(),
            hidden_dims: vec![8],
            output_dim: 1,
            activation: Activation::Relu,
        }],
        relations: vec![],
    };
    let model = UniMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        1,
        Some(4),
        false,
        1.0,
        4,
        4,
        &task_config,
        false,
        vb.pp("unimixer"),
    )
    .unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert!(out.contains_key("ctr"));
    assert_eq!(out["ctr"].dims(), &[3, 1]);
}

#[test]
fn test_unimixer_tokenizer_accepts_sequence_pooling() {
    use scale_rec::feats::config::PoolingStrategy;
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let device = Device::Cpu;
    let mut seq = FeatureSpec::new("seq".into(), 16, 4);
    seq.pooling = PoolingStrategy::First;
    let mut flat = FeatureSpec::new("flat".into(), 16, 2);
    flat.pooling = PoolingStrategy::Flatten;
    flat.seq_len = Some(2);
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &[seq, flat], 3, 2).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(
        "seq".into(),
        Tensor::from_slice(&[1u32, 2, 3, 4], (2, 2), &device).unwrap(),
    );
    inputs.insert(
        "flat".into(),
        Tensor::from_slice(&[1u32, 2, 3, 4], (2, 2), &device).unwrap(),
    );

    let out = tokenizer.forward(&inputs).unwrap();

    assert_eq!(out.dims(), &[2, 2, 3]);
}

#[test]
fn test_unimixer_rejects_invalid_temperature() {
    use scale_rec::models::unimixer::{model::UniMixerModel, tokenizer::FeatureTokenizer};

    let features = dummy_features();
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &features, 4, 2).unwrap();
    let task_config = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "ctr".into(),
            hidden_dims: vec![8],
            output_dim: 1,
            activation: Activation::Relu,
        }],
        relations: vec![],
    };
    let model = UniMixerModel::new(
        tokenizer,
        4,
        2,
        1,
        Some(4),
        false,
        1.0,
        4,
        4,
        &task_config,
        false,
        vb.pp("unimixer"),
    )
    .unwrap();

    let err = model
        .forward_with_temperature(&dummy_inputs(3), 0.0)
        .unwrap_err();
    assert!(err.to_string().contains("temperature must be > 0"));
}

#[test]
fn test_modelconfig_build_mmoe() {
    let mut params = serde_yaml::Mapping::new();
    params.insert("num_experts".into(), serde_yaml::Value::Number(2.into()));
    let mut expert_dims = serde_yaml::Sequence::new();
    expert_dims.push(serde_yaml::Value::Number(8.into()));
    params.insert(
        "expert_hidden_dims".into(),
        serde_yaml::Value::Sequence(expert_dims),
    );
    params.insert(
        "expert_output_dim".into(),
        serde_yaml::Value::Number(4.into()),
    );
    let mut tcs = serde_yaml::Sequence::new();
    let mut entry = serde_yaml::Mapping::new();
    entry.insert("name".into(), serde_yaml::Value::String("ctr".into()));
    let mut td = serde_yaml::Sequence::new();
    td.push(serde_yaml::Value::Number(4.into()));
    entry.insert("tower_dims".into(), serde_yaml::Value::Sequence(td));
    tcs.push(serde_yaml::Value::Mapping(entry));
    params.insert("task_configs".into(), serde_yaml::Value::Sequence(tcs));
    let cfg = ModelConfig {
        model_type: "mmoe".into(),
        params: serde_yaml::Value::Mapping(params),
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("ctr"));
}
