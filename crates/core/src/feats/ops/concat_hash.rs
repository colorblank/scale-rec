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
