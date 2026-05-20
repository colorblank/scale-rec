//! 特征哈希算子：DJB2 多种子哈希，与 Python 实现逐位一致。

use crate::feats::ops::{CustomOp, Fv};

/// 无状态特征哈希：拼接多输入 → k 个种子 DJB2 → k 个索引。
///
/// 每个种子产生一个 `[0, vocab_size)` 范围内的索引。多哈希降低碰撞率。
pub struct FeatureHash {
    vocab_size: u32,
    num_hashes: u32,
    separator: String,
}

impl FeatureHash {
    pub fn new(vocab_size: u32, num_hashes: u32, separator: String) -> Self {
        assert!(vocab_size > 0, "vocab_size must be positive");
        Self {
            vocab_size,
            num_hashes,
            separator,
        }
    }
}

impl CustomOp for FeatureHash {
    fn name(&self) -> &str {
        "FeatureHash"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let key = build_key(inputs, &self.separator);
        Ok(hash_multi(&key, self.num_hashes, self.vocab_size))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        if inputs.is_empty() || n_rows == 0 {
            return Ok(vec![]);
        }
        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            // 逐行拼接所有输入列
            let mut key = String::new();
            for (ci, col) in inputs.iter().enumerate() {
                if ci > 0 {
                    key.push_str(&self.separator);
                }
                if row < col.len() {
                    key.push_str(&col[row].to_string());
                }
            }
            results.push(hash_multi(&key, self.num_hashes, self.vocab_size));
        }
        Ok(results)
    }
}

// ── 公开工具函数 (供测试) ──

/// DJB2 带种子前缀的哈希，32 位回绕 —— 与 Python _djb2_seeded 完全一致。
pub fn djb2_seeded(key: &str, seed: u32) -> u32 {
    let mut h: u32 = 5381;
    // 种子前缀 (同 Python: for ch in str(seed))
    for ch in seed.to_string().bytes() {
        h = h.wrapping_mul(33).wrapping_add(ch as u32);
    }
    // 分隔符 '_'
    h = h.wrapping_mul(33).wrapping_add(b'_' as u32);
    // 键体
    for ch in key.bytes() {
        h = h.wrapping_mul(33).wrapping_add(ch as u32);
    }
    h & 0x7FFF_FFFF
}

// ── 内部 ──

fn build_key(inputs: &[Fv], sep: &str) -> String {
    let parts: Vec<String> = inputs.iter().map(|v| v.to_string()).collect();
    parts.join(sep)
}

fn hash_multi(key: &str, num_hashes: u32, vocab_size: u32) -> Fv {
    if num_hashes == 1 {
        Fv::Int((djb2_seeded(key, 0) % vocab_size) as i32)
    } else {
        let indices: Vec<i32> = (0..num_hashes)
            .map(|s| (djb2_seeded(key, s) % vocab_size) as i32)
            .collect();
        Fv::IntList(indices)
    }
}

// ── 测试 ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_djb2_matches_python() {
        // Python 参考输出 (已验证)
        assert_eq!(djb2_seeded("hello_world", 0), 1442432207);
        assert_eq!(djb2_seeded("hello_world", 1), 626576976);
        assert_eq!(djb2_seeded("hello_world", 4), 326494931);
        assert_eq!(djb2_seeded("author_617931428", 0), 1395292063);
        assert_eq!(djb2_seeded("", 0), 5861588);
        assert_eq!(djb2_seeded("", 99), 193441046);
    }

    #[test]
    fn test_single_hash() {
        let op = FeatureHash::new(1000, 1, "|".into());
        let result = op
            .process(&[Fv::Str("user_123".into()), Fv::Str("item_456".into())])
            .unwrap();
        if let Fv::Int(idx) = result {
            assert!(idx >= 0 && idx < 1000);
        } else {
            panic!("Expected Int");
        }
    }

    #[test]
    fn test_multi_hash() {
        let op = FeatureHash::new(1000, 4, "|".into());
        let result = op.process(&[Fv::Str("hello".into())]).unwrap();
        if let Fv::IntList(indices) = result {
            assert_eq!(indices.len(), 4);
            for &idx in &indices {
                assert!(idx >= 0 && idx < 1000);
            }
            // 4 个种子应产生不同值 (高概率)
            let unique: std::collections::HashSet<i32> = indices.into_iter().collect();
            assert!(unique.len() >= 3, "Expected at least 3 distinct hashes");
        } else {
            panic!("Expected IntList");
        }
    }

    #[test]
    fn test_deterministic() {
        let op = FeatureHash::new(500, 3, "_".into());
        let a = op.process(&[Fv::Str("test".into())]).unwrap();
        let b = op.process(&[Fv::Str("test".into())]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_inputs_different_output() {
        let op = FeatureHash::new(100000, 1, "|".into());
        let a = op.process(&[Fv::Str("abc".into())]).unwrap();
        let b = op.process(&[Fv::Str("abd".into())]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_empty_input() {
        let op = FeatureHash::new(100, 2, "|".into());
        let result = op.process(&[Fv::Str("".into())]).unwrap();
        if let Fv::IntList(indices) = result {
            assert_eq!(indices.len(), 2);
        } else {
            panic!("Expected IntList");
        }
    }

    #[test]
    fn test_batch() {
        let op = FeatureHash::new(1000, 2, "|".into());
        let col_a = vec![Fv::Str("x".into()), Fv::Str("y".into())];
        let col_b = vec![Fv::Str("1".into()), Fv::Str("2".into())];
        let results = op.process_batch(&[&col_a, &col_b], 2).unwrap();
        assert_eq!(results.len(), 2);
        // 逐行应与 process 一致
        let single_0 = op
            .process(&[Fv::Str("x".into()), Fv::Str("1".into())])
            .unwrap();
        let single_1 = op
            .process(&[Fv::Str("y".into()), Fv::Str("2".into())])
            .unwrap();
        assert_eq!(results[0], single_0);
        assert_eq!(results[1], single_1);
    }
}
