//! 字符串解析算子：结构化字符串分词与填充。
use super::{CustomOp, Fv};

/// 结构化字符串解析：按 sep1 分词、sep2 提取、填充/截断。
pub struct StringParser {
    sep1: String,
    sep2: String,
    key_index: usize,
    pad_len: usize,
    pad_val: String,
}

impl StringParser {
    /// 创建字符串解析算子。
    pub fn new(
        sep1: String,
        sep2: String,
        key_index: usize,
        pad_len: usize,
        pad_val: String,
    ) -> Self {
        Self {
            sep1,
            sep2,
            key_index,
            pad_len,
            pad_val,
        }
    }
}

/// 从 YAML params 创建 StringParser 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let sep1 = params
        .get("sep1")
        .and_then(|v| v.as_str())
        .unwrap_or("#")
        .to_string();
    let sep2 = params
        .get("sep2")
        .and_then(|v| v.as_str())
        .unwrap_or("|")
        .to_string();
    let key_index = params
        .get("key_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let pad_len = params.get("pad_len").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let pad_val = params
        .get("pad_val")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    Ok(Box::new(StringParser::new(
        sep1, sep2, key_index, pad_len, pad_val,
    )))
}

impl CustomOp for StringParser {
    fn name(&self) -> &str {
        "StringParser"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let s = match &inputs[0] {
            Fv::Str(s) => s.as_str(),
            _ => "",
        };
        let mut result: Vec<String> = if s.is_empty() {
            Vec::new()
        } else {
            s.split(&self.sep1)
                .filter_map(|item| {
                    let parts: Vec<&str> = item.split(&self.sep2).collect();
                    if self.key_index < parts.len() {
                        Some(parts[self.key_index].to_string())
                    } else {
                        None
                    }
                })
                .collect()
        };
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
            let mut result: Vec<String> = if s.is_empty() {
                Vec::new()
            } else {
                s.split(&self.sep1)
                    .filter_map(|item| {
                        let parts: Vec<&str> = item.split(&self.sep2).collect();
                        if self.key_index < parts.len() {
                            Some(parts[self.key_index].to_string())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            while result.len() < self.pad_len {
                result.push(self.pad_val.clone());
            }
            result.truncate(self.pad_len);
            results.push(Fv::StrList(result));
        }
        Ok(results)
    }
}
