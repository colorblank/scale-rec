use std::collections::HashMap;
use std::fs;
use scale_rec::feats::config::{FlowConfig, DType};
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
use scale_rec::feats::ops::Fv;
use serde_json::{json, Value};

fn fixture_path(name: &str) -> String {
    format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn json_to_fv_typed(value: &Value, dtype: &DType) -> FeatureValue {
    match dtype {
        DType::Int => match value {
            Value::Number(n) => Fv::Int(n.as_i64().unwrap_or(0) as i32),
            Value::String(s) => Fv::Int(s.parse().unwrap_or(0)),
            _ => Fv::Int(0),
        },
        DType::Float => match value {
            Value::Number(n) => Fv::Float(n.as_f64().unwrap_or(0.0) as f32),
            Value::String(s) => Fv::Float(s.parse().unwrap_or(0.0)),
            _ => Fv::Float(0.0),
        },
        DType::String | DType::Enum { .. } => match value {
            Value::String(s) => Fv::Str(s.clone()),
            _ => Fv::Str(value.to_string()),
        },
        DType::List { dtype: inner, .. } => {
            if let Value::Array(arr) = value {
                match inner.as_ref() {
                    DType::Int => Fv::IntList(
                        arr.iter()
                            .map(|v| match v {
                                Value::Number(n) => n.as_i64().unwrap_or(0) as i32,
                                Value::String(s) => s.parse().unwrap_or(0),
                                _ => 0,
                            })
                            .collect(),
                    ),
                    DType::Float => Fv::FloatList(
                        arr.iter()
                            .map(|v| match v {
                                Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
                                Value::String(s) => s.parse().unwrap_or(0.0),
                                _ => 0.0,
                            })
                            .collect(),
                    ),
                    _ => Fv::StrList(
                        arr.iter()
                            .map(|v| match v {
                                Value::String(s) => s.clone(),
                                _ => v.to_string(),
                            })
                            .collect(),
                    ),
                }
            } else {
                Fv::StrList(vec![])
            }
        }
    }
}

fn fv_to_json(value: &FeatureValue) -> Value {
    match value {
        Fv::Int(value) => json!(value),
        Fv::Float(value) => json!(format!("{:.5}", value).parse::<f64>().unwrap()),
        Fv::Str(value) => json!(value),
        Fv::IntList(values) => json!(values),
        Fv::FloatList(values) => json!(values
            .iter()
            .map(|&v| format!("{:.5}", v).parse::<f64>().unwrap())
            .collect::<Vec<_>>()),
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
                .filter(|(_, value)| !value.is_null())
                .map(|(key, value)| {
                    let source_def = dag.source_defs().get(key).expect("unknown source key");
                    (key.clone(), json_to_fv_typed(value, &source_def.dtype))
                })
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

    for (ri, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        if act != exp {
            println!("Row {} mismatch!", ri);
            for (key, act_val) in act {
                if let Some(exp_val) = exp.get(key) {
                    if act_val != exp_val {
                        println!("  Feature '{}':", key);
                        println!("    Actual:   {:?}", act_val);
                        println!("    Expected: {:?}", exp_val);
                    }
                } else {
                    println!("  Feature '{}' expected is missing", key);
                }
            }
            for key in exp.keys() {
                if !act.contains_key(key) {
                    println!("  Feature '{}' actual is missing", key);
                }
            }
        }
    }

    assert_eq!(actual, expected);
}
