//! 融合预处理算子：先解析为 token 序列，再逐 token hash。

use crate::feats::ops::{feature_hash::FeatureHash, CustomOp, Fv};
use serde_json::Value;

pub struct ParsedFeatureHash {
    parse_mode: String,
    key: Option<String>,
    pad_len: usize,
    pad_val: String,
    sep1: String,
    sep2: String,
    key_index: usize,
    sep: String,
    max_len: usize,
    inner: FeatureHash,
}

impl ParsedFeatureHash {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: u32,
        parse_mode: String,
        num_hashes: u32,
        separator: String,
        namespace: String,
        salt: String,
        version: String,
        key: Option<String>,
        sep1: String,
        sep2: String,
        key_index: usize,
        sep: String,
        max_len: usize,
        pad_len: usize,
        pad_val: String,
    ) -> Result<Self, String> {
        if num_hashes != 1 {
            return Err("ParsedFeatureHash only supports num_hashes=1".to_string());
        }
        Ok(Self {
            parse_mode,
            key,
            pad_len,
            pad_val,
            sep1,
            sep2,
            key_index,
            sep,
            max_len,
            inner: FeatureHash::with_scope(vocab_size, 1, separator, &namespace, &salt, &version)?,
        })
    }

    fn parse_tokens(&self, input: &Fv) -> Result<Vec<String>, String> {
        match self.parse_mode.as_str() {
            "json" => self.parse_json(input),
            "structured" => self.parse_structured(input),
            "structured_flat_split" => self.parse_structured_flat_split(input),
            "split" => Ok(self.normalize_max(match input {
                Fv::Str(s) => {
                    if s.is_empty() {
                        Vec::new()
                    } else {
                        s.split(&self.sep).map(|x| x.to_string()).collect()
                    }
                }
                _ => Vec::new(),
            })),
            "list_split" => self.parse_list_split(input),
            "flat_split" => self.parse_flat_split(input),
            other => Err(format!("unsupported ParsedFeatureHash mode: {}", other)),
        }
    }

    fn parse_json(&self, input: &Fv) -> Result<Vec<String>, String> {
        let s = match input {
            Fv::Str(s) => s.as_str(),
            _ => "",
        };
        let mut result = Vec::new();
        if !s.is_empty() {
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(s) {
                for item in arr {
                    if let Some(k) = &self.key {
                        if let Some(val) = item.get(k) {
                            if let Some(v_str) = val.as_str() {
                                result.push(v_str.to_string());
                            } else if let Some(b) = val.as_bool() {
                                result.push(if b { "True".into() } else { "False".into() });
                            } else {
                                result.push(val.to_string());
                            }
                        }
                    } else if let Some(v_str) = item.as_str() {
                        result.push(v_str.to_string());
                    } else if let Some(b) = item.as_bool() {
                        result.push(if b { "True".into() } else { "False".into() });
                    } else {
                        result.push(item.to_string());
                    }
                }
            }
        }
        Ok(self.normalize(result))
    }

    fn parse_structured(&self, input: &Fv) -> Result<Vec<String>, String> {
        let s = match input {
            Fv::Str(s) => s.as_str(),
            _ => "",
        };
        let mut result = Vec::new();
        if !s.is_empty() {
            for item in s.split(&self.sep1) {
                let parts: Vec<&str> = item.split(&self.sep2).collect();
                if self.key_index < parts.len() {
                    result.push(parts[self.key_index].to_string());
                }
            }
        }
        Ok(self.normalize(result))
    }

    fn parse_list_split(&self, input: &Fv) -> Result<Vec<String>, String> {
        let list = match input {
            Fv::StrList(l) => l,
            _ => return Err("ParsedFeatureHash list_split requires StrList input".into()),
        };
        let mut result = Vec::with_capacity(list.len());
        for item in list {
            let parts: Vec<&str> = item.split(&self.sep).collect();
            if self.key_index < parts.len() {
                result.push(parts[self.key_index].to_string());
            } else {
                result.push(String::new());
            }
        }
        Ok(self.normalize(result))
    }

    fn parse_flat_split(&self, input: &Fv) -> Result<Vec<String>, String> {
        let list = match input {
            Fv::StrList(l) => l,
            _ => return Err("ParsedFeatureHash flat_split requires StrList input".into()),
        };
        let mut result = Vec::new();
        for item in list {
            if !item.is_empty() {
                result.extend(item.split(&self.sep).map(|s| s.to_string()));
            }
        }
        Ok(self.normalize_max(result))
    }

    fn parse_structured_flat_split(&self, input: &Fv) -> Result<Vec<String>, String> {
        let s = match input {
            Fv::Str(s) => s.as_str(),
            _ => "",
        };
        let mut result = Vec::new();
        if !s.is_empty() {
            for item in s.split(&self.sep1) {
                let parts: Vec<&str> = item.split(&self.sep2).collect();
                if self.key_index < parts.len() {
                    let token = parts[self.key_index];
                    if !token.is_empty() {
                        result.extend(token.split(&self.sep).map(|s| s.to_string()));
                    }
                }
            }
        }
        Ok(self.normalize_max(result))
    }

    fn normalize(&self, mut parts: Vec<String>) -> Vec<String> {
        if self.pad_len == 0 {
            return parts;
        }
        if parts.len() > self.pad_len {
            parts.truncate(self.pad_len);
        }
        while parts.len() < self.pad_len {
            parts.push(self.pad_val.clone());
        }
        parts
    }

    fn normalize_max(&self, mut parts: Vec<String>) -> Vec<String> {
        if self.max_len == 0 {
            return parts;
        }
        if parts.len() > self.max_len {
            parts.truncate(self.max_len);
        }
        while parts.len() < self.max_len {
            parts.push(self.pad_val.clone());
        }
        parts
    }
}

impl CustomOp for ParsedFeatureHash {
    fn name(&self) -> &str {
        "ParsedFeatureHash"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let tokens = self.parse_tokens(&inputs[0])?;
        self.inner.process(&[Fv::StrList(tokens)])
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let tokens = self.parse_tokens(&col[i])?;
            results.push(self.inner.process(&[Fv::StrList(tokens)])?);
        }
        Ok(results)
    }
}
