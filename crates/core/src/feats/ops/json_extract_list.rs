//! JSON array extraction operator: parses JSON arrays/object arrays and extracts values.
use super::{CustomOp, Fv};
use serde_json::Value;

/// Extracts string values from a JSON array, with optional key-based extraction
/// from object arrays. Pads/truncates to a fixed length.
pub struct JsonExtractList {
    key: Option<String>,
    pad_len: usize,
    pad_val: String,
}

impl JsonExtractList {
    pub fn new(key: Option<String>, pad_len: usize, pad_val: String) -> Self {
        Self {
            key,
            pad_len,
            pad_val,
        }
    }

    fn extract_values(&self, s: &str) -> Vec<String> {
        let mut result = Vec::new();
        if let Ok(Value::Array(arr)) = serde_json::from_str(s) {
            for item in arr {
                if let Some(k) = &self.key {
                    if let Some(val) = item.get(k) {
                        if let Some(v_str) = val.as_str() {
                            result.push(v_str.to_string());
                        } else if let Some(b) = val.as_bool() {
                            result.push(if b {
                                "True".to_string()
                            } else {
                                "False".to_string()
                            });
                        } else {
                            result.push(val.to_string());
                        }
                    }
                } else if let Some(v_str) = item.as_str() {
                    result.push(v_str.to_string());
                } else if let Some(b) = item.as_bool() {
                    result.push(if b {
                        "True".to_string()
                    } else {
                        "False".to_string()
                    });
                } else {
                    result.push(item.to_string());
                }
            }
        }
        while result.len() < self.pad_len {
            result.push(self.pad_val.clone());
        }
        result.truncate(self.pad_len);
        result
    }
}

impl CustomOp for JsonExtractList {
    fn name(&self) -> &str {
        "JsonExtractList"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let s = match &inputs[0] {
            Fv::Str(s) => s.as_str(),
            _ => "",
        };
        Ok(Fv::StrList(self.extract_values(s)))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let s = match &col[i] {
                Fv::Str(s) => s.as_str(),
                _ => "",
            };
            results.push(Fv::StrList(self.extract_values(s)));
        }
        Ok(results)
    }
}

pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let key = params
        .get("key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let pad_len = params.get("pad_len").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let pad_val = params
        .get("pad_val")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(Box::new(JsonExtractList::new(key, pad_len, pad_val)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_extract_list() {
        let op_obj = JsonExtractList::new(Some("tag".into()), 2, "none".into());
        let input_obj = Fv::Str(
            "[{\"score\":0.99,\"tag\":\"invest\"}, {\"score\":0.5,\"tag\":\"finance\"}]".into(),
        );
        let res_obj = op_obj.process(&[input_obj]).unwrap();
        assert_eq!(
            res_obj,
            Fv::StrList(vec!["invest".into(), "finance".into()])
        );

        let op_str = JsonExtractList::new(None, 2, "none".into());
        let input_str = Fv::Str("[\"603538,17\"]".into());
        let res_str = op_str.process(&[input_str]).unwrap();
        assert_eq!(
            res_str,
            Fv::StrList(vec!["603538,17".into(), "none".into()])
        );

        let res_empty = op_str.process(&[Fv::Str("".into())]).unwrap();
        assert_eq!(res_empty, Fv::StrList(vec!["none".into(), "none".into()]));
    }

    #[test]
    fn test_json_extract_bool_and_number() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        let input = Fv::Str("[{\"tag\":true},{\"tag\":123.4}]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["True".into(), "123.4".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_key_missing() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        let input = Fv::Str("[{\"not_tag\":\"value\"}]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["none".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_invalid() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        let input = Fv::Str("invalid json strings { }".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["none".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_empty_string_value() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        let input = Fv::Str("[{\"tag\":\"\"}]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_empty_array() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        let input = Fv::Str("[]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["none".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_escaped_string() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        let input = Fv::Str("[{\"tag\":\"hello\\\"world\"}]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["hello\"world".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_no_key_with_objects() {
        let op = JsonExtractList::new(None, 3, "none".into());
        let input = Fv::Str("[\"a\",\"b\"]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["a".into(), "b".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_null_value_in_array() {
        let op = JsonExtractList::new(Some("tag".into()), 3, "none".into());
        // serde_json serialises null as "null" via to_string
        let input = Fv::Str("[{\"tag\":null},{\"tag\":\"real\"}]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(
            res,
            Fv::StrList(vec!["null".into(), "real".into(), "none".into()])
        );
    }

    #[test]
    fn test_json_extract_non_string_fv_input() {
        let op = JsonExtractList::new(Some("tag".into()), 2, "pad".into());
        let res = op.process(&[Fv::Int(123)]).unwrap();
        assert_eq!(res, Fv::StrList(vec!["pad".into(), "pad".into()]));
    }

    #[test]
    fn test_json_extract_pad_len_zero() {
        let op = JsonExtractList::new(Some("tag".into()), 0, "ignored".into());
        let input = Fv::Str("[{\"tag\":\"a\"},{\"tag\":\"b\"}]".into());
        let res = op.process(&[input]).unwrap();
        assert_eq!(res, Fv::StrList(vec![] as Vec<String>));
    }

    #[test]
    fn test_json_extract_batch() {
        let op = JsonExtractList::new(Some("tag".into()), 2, "none".into());
        let col = vec![
            Fv::Str("[{\"tag\":\"a\"}]".into()),
            Fv::Str("invalid".into()),
            Fv::Str("[{\"tag\":\"b\"},{\"tag\":\"c\"}]".into()),
        ];
        let res = op.process_batch(&[&col], 3).unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], Fv::StrList(vec!["a".into(), "none".into()]));
        assert_eq!(res[1], Fv::StrList(vec!["none".into(), "none".into()]));
        assert_eq!(res[2], Fv::StrList(vec!["b".into(), "c".into()]));
    }
}
