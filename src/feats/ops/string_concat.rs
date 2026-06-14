//! 字符串拼接算子：多输入 → 单字符串输出。

use crate::feats::ops::{CustomOp, Fv};

/// 将多路输入拼接为字符串。输入取值类型不限，输出为 `Fv::Str`。
pub struct StringConcat {
    separator: String,
}

impl StringConcat {
    /// 创建字符串拼接算子。
    pub fn new(separator: String) -> Self {
        Self { separator }
    }
}

impl CustomOp for StringConcat {
    fn name(&self) -> &str {
        "StringConcat"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let key: String = inputs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(&self.separator);
        Ok(Fv::Str(key))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        if n_rows == 0 {
            return Ok(vec![]);
        }
        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let key: String = inputs
                .iter()
                .map(|col| {
                    if row < col.len() {
                        col[row].to_string()
                    } else {
                        String::new()
                    }
                })
                .collect::<Vec<_>>()
                .join(&self.separator);
            results.push(Fv::Str(key));
        }
        Ok(results)
    }
}

pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let separator = params
        .get("separator")
        .and_then(|v| v.as_str())
        .unwrap_or("_")
        .to_string();
    Ok(Box::new(StringConcat::new(separator)))
}

// ── 测试 ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_input() {
        let op = StringConcat::new("_".into());
        let result = op.process(&[Fv::Str("hello".into())]).unwrap();
        assert_eq!(result, Fv::Str("hello".into()));
    }

    #[test]
    fn test_two_string_inputs() {
        let op = StringConcat::new("_".into());
        let result = op
            .process(&[Fv::Str("user".into()), Fv::Str("123".into())])
            .unwrap();
        assert_eq!(result, Fv::Str("user_123".into()));
    }

    #[test]
    fn test_int_and_str() {
        let op = StringConcat::new("|".into());
        let result = op.process(&[Fv::Int(42), Fv::Str("abc".into())]).unwrap();
        assert_eq!(result, Fv::Str("42|abc".into()));
    }

    #[test]
    fn test_custom_separator() {
        let op = StringConcat::new("#".into());
        let result = op
            .process(&[
                Fv::Str("a".into()),
                Fv::Str("b".into()),
                Fv::Str("c".into()),
            ])
            .unwrap();
        assert_eq!(result, Fv::Str("a#b#c".into()));
    }

    #[test]
    fn test_empty_input() {
        let op = StringConcat::new("_".into());
        let result = op.process(&[]).unwrap();
        assert_eq!(result, Fv::Str("".into()));
    }

    #[test]
    fn test_batch() {
        let op = StringConcat::new("_".into());
        let col_a = vec![Fv::Str("x".into()), Fv::Str("y".into())];
        let col_b = vec![Fv::Str("1".into()), Fv::Str("2".into())];
        let results = op.process_batch(&[&col_a, &col_b], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], Fv::Str("x_1".into()));
        assert_eq!(results[1], Fv::Str("y_2".into()));
    }
}
