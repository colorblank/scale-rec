//! 融合预处理算子：多输入拼接后直接 hash。

use crate::feats::ops::{feature_hash::FeatureHash, CustomOp, Fv};

/// 多输入字符串拼接后合并哈希。
pub struct ConcatHash {
    separator: String,
    inner: FeatureHash,
}

impl ConcatHash {
    #[allow(clippy::too_many_arguments)]
    /// 创建 ConcatHash 算子。
    pub fn new(
        vocab_size: u32,
        num_hashes: u32,
        separator: String,
        namespace: String,
        salt: String,
        version: String,
    ) -> Result<Self, String> {
        Ok(Self {
            separator: separator.clone(),
            inner: FeatureHash::with_scope(
                vocab_size, num_hashes, separator, &namespace, &salt, &version,
            )?,
        })
    }
}

fn push_fv(buf: &mut String, v: &Fv) {
    match v {
        Fv::Str(s) => buf.push_str(s),
        other => buf.push_str(&other.to_string()),
    }
}

/// 从 YAML params 创建 ConcatHash 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let vocab_size = params
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(1000) as u32;
    let num_hashes = params
        .get("num_hashes")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let separator = params
        .get("separator")
        .and_then(|v| v.as_str())
        .unwrap_or("_")
        .to_string();
    let namespace = params
        .get("namespace")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let salt = params
        .get("salt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let version = params
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    Ok(Box::new(ConcatHash::new(
        vocab_size, num_hashes, separator, namespace, salt, version,
    )?))
}

impl CustomOp for ConcatHash {
    fn name(&self) -> &str {
        "ConcatHash"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let mut key = String::new();
        for (i, v) in inputs.iter().enumerate() {
            if i > 0 {
                key.push_str(&self.separator);
            }
            push_fv(&mut key, v);
        }
        self.inner.process(&[Fv::Str(key)])
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        if n_rows == 0 {
            return Ok(vec![]);
        }
        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            let mut key = String::new();
            for (ci, col) in inputs.iter().enumerate() {
                if ci > 0 {
                    key.push_str(&self.separator);
                }
                if row < col.len() {
                    push_fv(&mut key, &col[row]);
                }
            }
            results.push(self.inner.process(&[Fv::Str(key)])?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op() -> ConcatHash {
        ConcatHash::new(100, 1, "_".into(), "".into(), "".into(), "".into()).unwrap()
    }

    fn assert_hash_in_range(res: Fv) {
        assert!(
            matches!(res, Fv::Int(v) if v >= 0 && v < 100),
            "expected Int in [0, 100), got {:?}",
            res
        );
    }

    #[test]
    fn test_concat_basic() {
        let op = make_op();
        let res = op
            .process(&[Fv::Str("a".into()), Fv::Str("b".into())])
            .unwrap();
        assert_hash_in_range(res);
    }

    #[test]
    fn test_concat_single_input() {
        let op = make_op();
        let res = op.process(&[Fv::Str("hello".into())]).unwrap();
        assert_hash_in_range(res);
    }

    #[test]
    fn test_concat_empty_inputs() {
        let op = make_op();
        let res = op.process(&[]).unwrap();
        assert_hash_in_range(res);
    }

    #[test]
    fn test_concat_float_input() {
        let op = make_op();
        let res = op.process(&[Fv::Float(3.14), Fv::Str("x".into())]).unwrap();
        assert_hash_in_range(res);
    }

    #[test]
    fn test_concat_empty_separator() {
        let op = ConcatHash::new(100, 1, "".into(), "".into(), "".into(), "".into()).unwrap();
        let res = op
            .process(&[Fv::Str("ab".into()), Fv::Str("cd".into())])
            .unwrap();
        assert_hash_in_range(res);
    }

    #[test]
    fn test_concat_batch_empty_returns_empty() {
        let op = make_op();
        let res = op.process_batch(&[], 0).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn test_concat_batch_with_wrong_type_fallback() {
        let op = make_op();
        let col_a = vec![Fv::IntList(vec![1, 2])];
        let col_b = vec![Fv::Str("x".into())];
        let res = op.process_batch(&[&col_a, &col_b], 1).unwrap();
        assert_eq!(res.len(), 1);
        assert_hash_in_range(res[0].clone());
    }
}
