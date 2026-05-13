use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

use scale_rec::layers::towers::{Activation, MultiTaskConfig, TowerConfig};
use scale_rec::models::{deepfm, esmm, lr, mmoe, Model, ModelConfig};

fn dummy_features() -> Vec<(String, usize, usize)> {
    vec![
        ("a".into(), 10, 4),
        ("b".into(), 5, 4),
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
    let model =
        mmoe::MMoE::new(vb(), &dummy_features(), &[8], 2, &[8], 4, &task_cfgs).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out["ctr"].dims(), &[3, 1]);
    assert_eq!(out["cvr"].dims(), &[3, 1]);
}

#[test]
fn test_esmm_forward_shape() {
    let model = esmm::ESMM::new(vb(), &dummy_features(), &[8], &[4], &[4]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out["ctr"].dims(), &[3, 1]);
    assert_eq!(out["cvr"].dims(), &[3, 1]);
    assert_eq!(out["ctcvr"].dims(), &[3, 1]);
}

#[test]
fn test_modelconfig_build_lr() {
    let cfg = ModelConfig::LR;
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("pred"));
}

#[test]
fn test_modelconfig_build_deepfm() {
    let cfg = ModelConfig::DeepFM {
        fm_k: 8,
        deep_hidden_dims: vec![4],
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("pred"));
}
