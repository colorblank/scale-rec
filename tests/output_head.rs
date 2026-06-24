use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::models::output_contract::OutputContract;
use scale_rec::models::output_head::OutputHead;
use scale_rec::models::OutputKind;

fn esmm_contract() -> OutputContract {
    let cases: serde_yaml::Value =
        serde_yaml::from_str(include_str!("fixtures/output_contract_cases.yaml")).unwrap();
    serde_yaml::from_value(cases["cases"][0]["contract"].clone()).unwrap()
}

fn regression_contract() -> OutputContract {
    serde_yaml::from_str(
        r#"
version: 1
graph:
  towers:
    - {name: left, kind: regression}
    - {name: right, kind: regression}
  relations:
    - {name: sum, op: add, inputs: [left, right]}
    - {name: value, op: identity, inputs: [sum]}
objectives: []
metrics: []
outputs:
  - {name: prediction, source: value}
"#,
    )
    .unwrap()
}

#[test]
fn output_head_executes_probability_graph_and_public_projection() {
    let contract = esmm_contract();
    let mut dims = HashMap::new();
    dims.insert("shared".to_string(), 4);
    let varmap = VarMap::new();
    let head = OutputHead::new(
        &contract,
        &dims,
        VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu),
    )
    .unwrap();
    let mut representations = HashMap::new();
    representations.insert(
        "shared".to_string(),
        Tensor::zeros((2, 4), DType::F32, &Device::Cpu).unwrap(),
    );

    let execution = head.forward(&representations).unwrap();
    let click_logit = execution.nodes.tensor("click_logit").unwrap();
    let cvr_logit = execution.nodes.tensor("cvr_logit").unwrap();
    let expected = candle_nn::ops::sigmoid(click_logit)
        .unwrap()
        .mul(&candle_nn::ops::sigmoid(cvr_logit).unwrap())
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    let actual = execution
        .nodes
        .tensor("ctcvr_prob")
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(
        execution.outputs.get("ctcvr").unwrap().kind,
        OutputKind::Probability
    );
    assert!(execution.outputs.contains_key("ctr"));
    assert!(!execution.outputs.contains_key("click_logit"));
}

#[test]
fn output_head_rejects_unknown_backbone_representation() {
    let contract = esmm_contract();
    let error = OutputHead::new(
        &contract,
        &HashMap::new(),
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu),
    )
    .err()
    .expect("unknown representation must fail");

    assert!(error
        .to_string()
        .contains("unknown representation 'shared'"));
}

#[test]
fn output_head_executes_regression_add_and_identity() {
    let contract = regression_contract();
    let mut dims = HashMap::new();
    dims.insert("shared".to_string(), 2);
    let head = OutputHead::new(
        &contract,
        &dims,
        VarBuilder::from_varmap(&VarMap::new(), DType::F32, &Device::Cpu),
    )
    .unwrap();
    let mut representations = HashMap::new();
    representations.insert(
        "shared".to_string(),
        Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap(),
    );

    let execution = head.forward(&representations).unwrap();
    let expected = execution
        .nodes
        .tensor("left")
        .unwrap()
        .broadcast_add(execution.nodes.tensor("right").unwrap())
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    assert_eq!(
        execution
            .outputs
            .tensor("prediction")
            .unwrap()
            .to_vec2::<f32>()
            .unwrap(),
        expected
    );
    assert_eq!(
        execution.outputs.get("prediction").unwrap().kind,
        OutputKind::Regression
    );
}
