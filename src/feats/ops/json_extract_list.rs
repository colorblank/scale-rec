//! JSON 数组提取算子：解析 JSON 数组/对象数组并提取内容。
use super::{CustomOp, Fv};
use serde_json::Value;

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
        let mut result = Vec::new();
        if !s.is_empty() {
            if let Ok(Value::Array(arr)) = serde_json::from_str(s) {
                for item in arr {
                    if let Some(k) = &self.key {
                        if let Some(val) = item.get(k) {
                            if let Some(v_str) = val.as_str() {
                                result.push(v_str.to_string());
                            } else if let Some(b) = val.as_bool() {
                                result.push(if b { "True".to_string() } else { "False".to_string() });
                            } else {
                                // Fallback to raw string representation (e.g. for numbers)
                                result.push(val.to_string());
                            }
                        }
                    } else {
                        if let Some(v_str) = item.as_str() {
                            result.push(v_str.to_string());
                        } else if let Some(b) = item.as_bool() {
                            result.push(if b { "True".to_string() } else { "False".to_string() });
                        } else {
                            result.push(item.to_string());
                        }
                    }
                }
            }
        }
        while result.len() < self.pad_len {
            result.push(self.pad_val.clone());
        }
        result.truncate(self.pad_len);
        Ok(Fv::StrList(result))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let s = match &col[i] {
                Fv::Str(s) => s.as_str(),
                _ => "",
            };
            let mut result = Vec::new();
            if !s.is_empty() {
                if let Ok(Value::Array(arr)) = serde_json::from_str(s) {
                    for item in arr {
                        if let Some(k) = &self.key {
                            if let Some(val) = item.get(k) {
                                if let Some(v_str) = val.as_str() {
                                    result.push(v_str.to_string());
                                } else if let Some(b) = val.as_bool() {
                                    result.push(if b { "True".to_string() } else { "False".to_string() });
                                } else {
                                    result.push(val.to_string());
                                }
                            }
                        } else {
                            if let Some(v_str) = item.as_str() {
                                result.push(v_str.to_string());
                            } else if let Some(b) = item.as_bool() {
                                result.push(if b { "True".to_string() } else { "False".to_string() });
                            } else {
                                    result.push(item.to_string());
                            }
                        }
                    }
                }
            }
            while result.len() < self.pad_len {
                result.push(self.pad_val.clone());
            }
            result.truncate(self.pad_len);
            results.push(Fv::StrList(result));
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_extract_list() {
        // Test extracting from object array
        let op_obj = JsonExtractList::new(Some("tag".into()), 2, "none".into());
        let input_obj = Fv::Str(
            "[{\"score\":0.99,\"tag\":\"invest\"}, {\"score\":0.5,\"tag\":\"finance\"}]".into(),
        );
        let res_obj = op_obj.process(&[input_obj]).unwrap();
        assert_eq!(
            res_obj,
            Fv::StrList(vec!["invest".into(), "finance".into()])
        );

        // Test extracting from simple string array
        let op_str = JsonExtractList::new(None, 2, "none".into());
        let input_str = Fv::Str("[\"603538,17\"]".into());
        let res_str = op_str.process(&[input_str]).unwrap();
        assert_eq!(
            res_str,
            Fv::StrList(vec!["603538,17".into(), "none".into()])
        );

        // Test empty string
        let res_empty = op_str.process(&[Fv::Str("".into())]).unwrap();
        assert_eq!(res_empty, Fv::StrList(vec!["none".into(), "none".into()]));
    }
}
