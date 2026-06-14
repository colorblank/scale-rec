//! 字符串分割算子：单字符串 → 按分隔符切分为定长列表。

use crate::feats::ops::{CustomOp, Fv};

/// 将输入字符串按分隔符切分，截断/填充到定长。
pub struct Split {
    sep: String,
    max_len: usize,
    pad_val: String,
}

impl Split {
    /// 创建字符串分割算子。
    pub fn new(sep: String, max_len: usize, pad_val: String) -> Self {
        Self {
            sep,
            max_len,
            pad_val,
        }
    }

    fn normalize(&self, mut parts: Vec<String>) -> Vec<String> {
        if self.max_len == 0 {
            return parts;
        }
        parts.truncate(self.max_len);
        while parts.len() < self.max_len {
            parts.push(self.pad_val.clone());
        }
        parts
    }
}

impl CustomOp for Split {
    fn name(&self) -> &str {
        "Split"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let s = match &inputs[0] {
            Fv::Str(s) => s.as_str(),
            _ => "",
        };
        let parts: Vec<String> = if s.is_empty() {
            Vec::new()
        } else {
            s.split(&self.sep).map(|p| p.to_string()).collect()
        };
        Ok(Fv::StrList(self.normalize(parts)))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        if n_rows == 0 {
            return Ok(vec![]);
        }
        let col = inputs.first().map(|c| *c).unwrap_or(&[]);
        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let s = if row < col.len() {
                match &col[row] {
                    Fv::Str(s) => s.clone(),
                    _ => String::new(),
                }
            } else {
                String::new()
            };
            let parts: Vec<String> = if s.is_empty() {
                Vec::new()
            } else {
                s.split(&self.sep).map(|p| p.to_string()).collect()
            };
            results.push(Fv::StrList(self.normalize(parts)));
        }
        Ok(results)
    }
}

pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let sep = params
        .get("sep")
        .and_then(|v| v.as_str())
        .unwrap_or("|")
        .to_string();
    let max_len = params.get("max_len").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let pad_val = params
        .get("pad_val")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(Box::new(Split::new(sep, max_len, pad_val)))
}

// ── 测试 ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_split() {
        let op = Split::new("|".into(), 0, "".into());
        let result = op.process(&[Fv::Str("a|b|c".into())]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["a".into(), "b".into(), "c".into()])
        );
    }

    #[test]
    fn test_single_element() {
        let op = Split::new("|".into(), 0, "".into());
        let result = op.process(&[Fv::Str("hello".into())]).unwrap();
        assert_eq!(result, Fv::StrList(vec!["hello".into()]));
    }

    #[test]
    fn test_empty_string() {
        let op = Split::new("|".into(), 0, "".into());
        let result = op.process(&[Fv::Str("".into())]).unwrap();
        assert_eq!(result, Fv::StrList(Vec::<String>::new()));
    }

    #[test]
    fn test_comma_separator() {
        let op = Split::new(",".into(), 0, "".into());
        let result = op.process(&[Fv::Str("x,y,z".into())]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["x".into(), "y".into(), "z".into()])
        );
    }

    #[test]
    fn test_truncate() {
        let op = Split::new("|".into(), 2, "".into());
        let result = op.process(&[Fv::Str("a|b|c".into())]).unwrap();
        assert_eq!(result, Fv::StrList(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_pad() {
        let op = Split::new("|".into(), 4, "none".into());
        let result = op.process(&[Fv::Str("a|b".into())]).unwrap();
        assert_eq!(
            result,
            Fv::StrList(vec!["a".into(), "b".into(), "none".into(), "none".into()])
        );
    }

    #[test]
    fn test_batch() {
        let op = Split::new("|".into(), 0, "".into());
        let col = vec![Fv::Str("a|b".into()), Fv::Str("c|d|e".into())];
        let results = op.process_batch(&[&col], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Fv::StrList(vec!["a".into(), "b".into()]));
        assert_eq!(
            results[1],
            Fv::StrList(vec!["c".into(), "d".into(), "e".into()])
        );
    }

    #[test]
    fn test_batch_truncate_pad() {
        let op = Split::new("|".into(), 2, "".into());
        let col = vec![Fv::Str("a".into()), Fv::Str("x|y|z".into())];
        let results = op.process_batch(&[&col], 2).unwrap();
        assert_eq!(results[0], Fv::StrList(vec!["a".into(), "".into()]));
        assert_eq!(results[1], Fv::StrList(vec!["x".into(), "y".into()]));
    }
}
