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

    fn parse_str(&self, s: &str) -> Vec<String> {
        let mut result = Vec::with_capacity(self.pad_len);
        if !s.is_empty() {
            for item in s.split(&self.sep1) {
                if result.len() >= self.pad_len {
                    break;
                }
                if let Some(part) = item.split(&self.sep2).nth(self.key_index) {
                    result.push(part.to_string());
                }
            }
        }
        while result.len() < self.pad_len {
            result.push(self.pad_val.clone());
        }
        result
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
        Ok(Fv::StrList(self.parse_str(s)))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let s = match &col[i] {
                Fv::Str(s) => s.as_str(),
                _ => "",
            };
            results.push(Fv::StrList(self.parse_str(s)));
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_key_index_and_pads() {
        let op = StringParser::new("|".into(), "#".into(), 0, 3, "unknown".into());
        let result = op.process(&[Fv::Str("a#1|b#2".into())]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["a".into(), "b".into(), "unknown".into()])
        );
    }

    #[test]
    fn truncates_after_pad_len() {
        let op = StringParser::new("|".into(), "#".into(), 0, 2, "unknown".into());
        let result = op.process(&[Fv::Str("a#1|b#2|c#3".into())]).unwrap();
        assert_eq!(result, Fv::StrList(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn skips_items_missing_key_index() {
        let op = StringParser::new("|".into(), "#".into(), 1, 2, "unknown".into());
        let result = op.process(&[Fv::Str("a#1|b".into())]).unwrap();
        assert_eq!(result, Fv::StrList(vec!["1".into(), "unknown".into()]));
    }

    #[test]
    fn empty_input_string() {
        let op = StringParser::new("|".into(), "#".into(), 0, 3, "pad".into());
        let result = op.process(&[Fv::Str("".into())]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["pad".into(), "pad".into(), "pad".into()])
        );
    }

    #[test]
    fn non_string_input_fallback() {
        let op = StringParser::new("|".into(), "#".into(), 0, 2, "x".into());
        let result = op.process(&[Fv::Int(42)]).unwrap();
        assert_eq!(result, Fv::StrList(vec!["x".into(), "x".into()]));
    }

    #[test]
    fn pad_len_zero() {
        let op = StringParser::new("|".into(), "#".into(), 0, 0, "pad".into());
        let result = op.process(&[Fv::Str("a|b|c".into())]).unwrap();
        assert_eq!(result, Fv::StrList(vec![] as Vec<String>));
    }
}
