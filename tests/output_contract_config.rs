use scale_rec::models::{output_contract::OutputContract, ModelConfig};

fn native_contract() -> serde_yaml::Value {
    serde_yaml::from_str(
        r#"
version: 1
graph:
  towers:
    - {name: score, kind: score}
  relations: []
objectives: []
metrics: []
outputs:
  - {name: score, source: score}
"#,
    )
    .unwrap()
}

#[test]
fn model_config_accepts_native_contract_without_legacy_task_config() {
    let mut params = serde_yaml::Mapping::new();
    params.insert("output_contract".into(), native_contract());
    let config = ModelConfig {
        model_type: "rankmixer".into(),
        params: serde_yaml::Value::Mapping(params),
    };

    let contract: OutputContract =
        serde_yaml::from_value(config.params["output_contract"].clone()).unwrap();
    contract.validate(None).unwrap();
}

#[test]
fn model_config_rejects_mixed_native_and_legacy_contracts_before_build() {
    let mut params = serde_yaml::Mapping::new();
    params.insert("output_contract".into(), native_contract());
    params.insert(
        "task_config".into(),
        serde_yaml::from_str("towers: []").unwrap(),
    );
    let config = ModelConfig {
        model_type: "rankmixer".into(),
        params: serde_yaml::Value::Mapping(params),
    };

    let error = config
        .build(
            candle_nn::VarBuilder::from_varmap(
                &candle_nn::VarMap::new(),
                candle_core::DType::F32,
                &candle_core::Device::Cpu,
            ),
            &[],
            None,
        )
        .err()
        .expect("mixed contract must fail");

    assert!(error.to_string().contains("cannot be combined"));
}
