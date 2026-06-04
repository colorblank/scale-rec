//! 融合预处理算子：多输入拼接后直接 hash。

use crate::feats::ops::{feature_hash::FeatureHash, CustomOp, Fv};

pub struct ConcatHash {
    separator: String,
    inner: FeatureHash,
}

impl ConcatHash {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab_size: u32,
        num_hashes: u32,
        separator: String,
        namespace: String,
        salt: String,
        version: String,
    ) -> Self {
        Self {
            separator: separator.clone(),
            inner: FeatureHash::with_scope(
                vocab_size,
                num_hashes,
                separator,
                &namespace,
                &salt,
                &version,
            ),
        }
    }
}

impl CustomOp for ConcatHash {
    fn name(&self) -> &str {
        "ConcatHash"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let key: String = inputs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(&self.separator);
        self.inner.process(&[Fv::Str(key)])
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
            results.push(self.inner.process(&[Fv::Str(key)])?);
        }
        Ok(results)
    }
}
