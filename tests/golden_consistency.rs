use std::collections::HashMap;
use std::fs;

use scale_rec::feats::config::FlowConfig;
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
use scale_rec::feats::ops::Fv;
use serde_json::{json, Value};

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn json_to_fv(value: &Value) -> FeatureValue {
    match value {
        Value::Number(number) => number
            .as_i64()
            .map(|n| Fv::Int(n as i32))
            .unwrap_or_else(|| Fv::Float(number.as_f64().unwrap_or_default() as f32)),
        Value::String(value) => Fv::Str(value.clone()),
        Value::Array(values) => Fv::StrList(
            values
                .iter()
                .map(|value| match value {
                    Value::String(value) => value.clone(),
                    _ => value.to_string(),
                })
                .collect(),
        ),
        _ => Fv::Str(value.to_string()),
    }
}

fn fv_to_json(value: &FeatureValue) -> Value {
    match value {
        Fv::Int(value) => json!(value),
        Fv::Float(value) => json!(value),
        Fv::Str(value) => json!(value),
        Fv::IntList(values) => json!(values),
        Fv::FloatList(values) => json!(values),
        Fv::StrList(values) => json!(values),
    }
}

#[test]
fn rust_dag_matches_golden_fixture() {
    let yaml = fs::read_to_string(fixture_path("golden_feature_config.yaml")).unwrap();
    let config = FlowConfig::from_yaml(&yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();
    let rows = fs::read_to_string(fixture_path("golden_rows.jsonl")).unwrap();
    let expected: Vec<HashMap<String, Value>> =
        serde_json::from_str(&fs::read_to_string(fixture_path("golden_expected.json")).unwrap())
            .unwrap();

    let actual: Vec<HashMap<String, Value>> = rows
        .lines()
        .zip(expected.iter())
        .map(|(line, expected_row)| {
            let raw_json: HashMap<String, Value> = serde_json::from_str(line).unwrap();
            let raw: HashMap<String, FeatureValue> = raw_json
                .iter()
                .map(|(key, value)| (key.clone(), json_to_fv(value)))
                .collect();
            let result = dag.execute(&raw).unwrap();
            expected_row
                .keys()
                .map(|name| {
                    (
                        name.clone(),
                        fv_to_json(result.features.get(name).expect("missing golden feature")),
                    )
                })
                .collect()
        })
        .collect();

    assert_eq!(actual, expected);
}
