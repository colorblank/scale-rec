//! 特征哈希算子：DJB2 多种子哈希，带缓存加速，与 Python 实现逐位一致。

use crate::feats::ops::{CustomOp, Fv};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use tracing::warn;

/// 特征哈希算子，内置线程安全缓存。
///
/// 缓存命中时跳过 DJB2 计算，直接从 HashMap 返回已哈希索引。
/// 推荐场景中高频特征（用户 ID、热门物品 ID）重复出现，缓存可显著减少计算。
pub struct FeatureHash {
    vocab_size: u32,
    num_hashes: u32,
    separator: String,
    hash_prefix: Option<String>,
    cache: RwLock<HashMap<String, Fv>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl FeatureHash {
    /// 创建特征哈希算子。
    pub fn new(vocab_size: u32, num_hashes: u32, separator: String) -> Result<Self, String> {
        Self::with_scope(vocab_size, num_hashes, separator, "", "", "")
    }

    /// 创建带命名空间/盐/版本的特征哈希算子。
    pub fn with_scope(
        vocab_size: u32,
        num_hashes: u32,
        separator: String,
        namespace: &str,
        salt: &str,
        version: &str,
    ) -> Result<Self, String> {
        if vocab_size == 0 {
            return Err("vocab_size must be positive".to_string());
        }
        let hash_prefix = build_hash_prefix(namespace, salt, version);
        Ok(Self {
            vocab_size,
            num_hashes,
            separator,
            hash_prefix,
            cache: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    /// 缓存命中次数（自创建以来的累计值）。
    pub fn cache_hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// 缓存未命中次数。
    pub fn cache_misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// 缓存条目数。
    pub fn cache_size(&self) -> usize {
        match self.cache.read() {
            Ok(cache) => cache.len(),
            Err(e) => {
                warn!("FeatureHash cache lock poisoned while reading size: {}", e);
                0
            }
        }
    }
}

impl CustomOp for FeatureHash {
    fn name(&self) -> &str {
        "FeatureHash"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        // 检测 list 输入 → 逐元素 hash
        let mut elems: Vec<String> = Vec::new();
        let mut has_list = false;
        for v in inputs {
            match v {
                Fv::StrList(list) => {
                    has_list = true;
                    elems.extend(list.iter().cloned());
                }
                Fv::IntList(list) => {
                    has_list = true;
                    elems.extend(list.iter().map(|i| i.to_string()));
                }
                Fv::FloatList(list) => {
                    has_list = true;
                    elems.extend(list.iter().map(|f| f.to_string()));
                }
                other => elems.push(other.to_string()),
            }
        }
        if has_list {
            let indices: Vec<i32> = elems.iter().map(|s| self.hash_one(s, 0)).collect();
            return Ok(Fv::IntList(indices));
        }
        let key = build_key(inputs, &self.separator);
        self.cached_or_compute(&key)
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        if inputs.is_empty() || n_rows == 0 {
            return Ok(vec![]);
        }
        let mut has_list_row = false;
        let mut has_scalar_row = false;
        for row in 0..n_rows {
            let row_has_list = inputs.iter().any(|col| {
                row < col.len()
                    && matches!(col[row], Fv::StrList(_) | Fv::IntList(_) | Fv::FloatList(_))
            });
            has_list_row |= row_has_list;
            has_scalar_row |= !row_has_list;
            if has_list_row && has_scalar_row {
                return Err(
                    "mixed scalar/list rows are not supported in FeatureHash batch".to_string(),
                );
            }
        }

        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        if has_list_row {
            if inputs.len() == 1 {
                let col = inputs[0];
                for value in col.iter().take(n_rows) {
                    let indices: Vec<i32> = match value {
                        Fv::StrList(list) => list.iter().map(|s| self.hash_one(s, 0)).collect(),
                        Fv::IntList(list) => list
                            .iter()
                            .map(|i| self.hash_one(&i.to_string(), 0))
                            .collect(),
                        Fv::FloatList(list) => list
                            .iter()
                            .map(|f| self.hash_one(&f.to_string(), 0))
                            .collect(),
                        other => vec![self.hash_one(&other.to_string(), 0)],
                    };
                    results.push(Fv::IntList(indices));
                }
                return Ok(results);
            }
            for row in 0..n_rows {
                let mut elems: Vec<String> = Vec::new();
                for col in inputs.iter() {
                    if row < col.len() {
                        match &col[row] {
                            Fv::StrList(list) => elems.extend(list.iter().cloned()),
                            Fv::IntList(list) => elems.extend(list.iter().map(|i| i.to_string())),
                            Fv::FloatList(list) => elems.extend(list.iter().map(|f| f.to_string())),
                            other => {
                                elems.push(other.to_string());
                            }
                        }
                    }
                }
                let indices: Vec<i32> = elems.iter().map(|s| self.hash_one(s, 0)).collect();
                results.push(Fv::IntList(indices));
            }
            return Ok(results);
        }
        // 标准路径：join → hash
        let mut cache = self
            .cache
            .write()
            .map_err(|e| format!("FeatureHash cache lock poisoned: {}", e))?;
        if inputs.len() == 1 {
            let col = inputs[0];
            for value in col.iter().take(n_rows) {
                match value {
                    Fv::Str(key) => {
                        if let Some(cached) = cache.get(key.as_str()) {
                            self.hits.fetch_add(1, Ordering::Relaxed);
                            results.push(cached.clone());
                        } else {
                            let val = self.hash_multi(key);
                            cache.insert(key.clone(), val.clone());
                            self.misses.fetch_add(1, Ordering::Relaxed);
                            results.push(val);
                        }
                    }
                    other => {
                        let key = other.to_string();
                        if let Some(cached) = cache.get(&key) {
                            self.hits.fetch_add(1, Ordering::Relaxed);
                            results.push(cached.clone());
                        } else {
                            let val = self.hash_multi(&key);
                            cache.insert(key, val.clone());
                            self.misses.fetch_add(1, Ordering::Relaxed);
                            results.push(val);
                        }
                    }
                }
            }
        } else {
            for row in 0..n_rows {
                let key = build_row_key(inputs, row, &self.separator);
                if let Some(cached) = cache.get(&key) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    results.push(cached.clone());
                } else {
                    let val = self.hash_multi(&key);
                    cache.insert(key, val.clone());
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    results.push(val);
                }
            }
        }
        Ok(results)
    }
}

impl FeatureHash {
    fn cached_or_compute(&self, key: &str) -> Result<Fv, String> {
        {
            let cache = self
                .cache
                .read()
                .map_err(|e| format!("FeatureHash cache lock poisoned: {}", e))?;
            if let Some(cached) = cache.get(key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached.clone());
            }
        }
        let result = self.hash_multi(key);
        {
            let mut cache = self
                .cache
                .write()
                .map_err(|e| format!("FeatureHash cache lock poisoned: {}", e))?;
            cache.insert(key.to_string(), result.clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }

    fn hash_one(&self, key: &str, seed: u32) -> i32 {
        (djb2_seeded_with_prefix(self.hash_prefix.as_deref(), key, seed) % self.vocab_size) as i32
    }

    fn hash_multi(&self, key: &str) -> Fv {
        if self.num_hashes == 1 {
            Fv::Int(self.hash_one(key, 0))
        } else {
            let indices: Vec<i32> = (0..self.num_hashes)
                .map(|seed| self.hash_one(key, seed))
                .collect();
            Fv::IntList(indices)
        }
    }
}

/// 从 YAML params 创建 FeatureHash 算子。
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
        .unwrap_or("|")
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
    Ok(Box::new(FeatureHash::with_scope(
        vocab_size, num_hashes, separator, &namespace, &salt, &version,
    )?))
}

// ── 公开工具函数 (供测试) ──

/// DJB2 带种子前缀的哈希，32 位回绕 —— 与 Python _djb2_seeded 完全一致。
pub fn djb2_seeded(key: &str, seed: u32) -> u32 {
    djb2_seeded_with_prefix(None, key, seed)
}

fn djb2_seeded_with_prefix(prefix: Option<&str>, key: &str, seed: u32) -> u32 {
    let mut h: u32 = 5381;
    for ch in seed.to_string().bytes() {
        h = h.wrapping_mul(33).wrapping_add(ch as u32);
    }
    h = h.wrapping_mul(33).wrapping_add(b'_' as u32);
    if let Some(prefix) = prefix {
        for ch in prefix.bytes() {
            h = h.wrapping_mul(33).wrapping_add(ch as u32);
        }
    }
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

fn build_hash_prefix(namespace: &str, salt: &str, version: &str) -> Option<String> {
    let parts: Vec<&str> = [namespace, salt, version]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(format!("{}::", parts.join("::")))
    }
}

/// 按行构建 key，避免为每行分配 Vec<String>。
fn build_row_key(inputs: &[&[Fv]], row: usize, sep: &str) -> String {
    let mut key = String::new();
    for (ci, col) in inputs.iter().enumerate() {
        if ci > 0 {
            key.push_str(sep);
        }
        if row < col.len() {
            key.push_str(&col[row].to_string());
        }
    }
    key
}

// ── 测试 ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_djb2_matches_python() {
        assert_eq!(djb2_seeded("hello_world", 0), 1442432207);
        assert_eq!(djb2_seeded("hello_world", 1), 626576976);
        assert_eq!(djb2_seeded("hello_world", 4), 326494931);
        assert_eq!(djb2_seeded("author_617931428", 0), 1395292063);
        assert_eq!(djb2_seeded("", 0), 5861588);
        assert_eq!(djb2_seeded("", 99), 193441046);
    }

    #[test]
    fn test_single_hash() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
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
        let op = FeatureHash::new(1000, 4, "|".into()).unwrap();
        let result = op.process(&[Fv::Str("hello".into())]).unwrap();
        if let Fv::IntList(indices) = result {
            assert_eq!(indices.len(), 4);
            for &idx in &indices {
                assert!(idx >= 0 && idx < 1000);
            }
            let unique: std::collections::HashSet<i32> = indices.into_iter().collect();
            assert!(unique.len() >= 3, "Expected at least 3 distinct hashes");
        } else {
            panic!("Expected IntList");
        }
    }

    #[test]
    fn test_deterministic() {
        let op = FeatureHash::new(500, 3, "_".into()).unwrap();
        let a = op.process(&[Fv::Str("test".into())]).unwrap();
        let b = op.process(&[Fv::Str("test".into())]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_inputs_different_output() {
        let op = FeatureHash::new(100000, 1, "|".into()).unwrap();
        let a = op.process(&[Fv::Str("abc".into())]).unwrap();
        let b = op.process(&[Fv::Str("abd".into())]).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_scope_changes_output_without_affecting_default() {
        let default_op = FeatureHash::new(1_000_000, 1, "|".into()).unwrap();
        let scoped_op =
            FeatureHash::with_scope(1_000_000, 1, "|".into(), "user_id", "salt", "v2").unwrap();
        assert_eq!(
            default_op.process(&[Fv::Str("abc".into())]).unwrap(),
            FeatureHash::with_scope(1_000_000, 1, "|".into(), "", "", "")
                .unwrap()
                .process(&[Fv::Str("abc".into())])
                .unwrap()
        );
        assert_ne!(
            default_op.process(&[Fv::Str("abc".into())]).unwrap(),
            scoped_op.process(&[Fv::Str("abc".into())]).unwrap()
        );
    }

    #[test]
    fn test_empty_input() {
        let op = FeatureHash::new(100, 2, "|".into()).unwrap();
        let result = op.process(&[Fv::Str("".into())]).unwrap();
        if let Fv::IntList(indices) = result {
            assert_eq!(indices.len(), 2);
        } else {
            panic!("Expected IntList");
        }
    }

    #[test]
    fn test_cache_hit() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        // 两次相同输入
        let a = op.process(&[Fv::Str("cached_key".into())]).unwrap();
        let b = op.process(&[Fv::Str("cached_key".into())]).unwrap();
        assert_eq!(a, b);
        assert!(op.cache_hits() >= 1);
        assert_eq!(op.cache_size(), 1);
    }

    #[test]
    fn test_cache_batch() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let col = vec![
            Fv::Str("dup".into()),
            Fv::Str("uniq".into()),
            Fv::Str("dup".into()), // 重复
        ];
        let results = op.process_batch(&[&col], 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], results[2]); // 第一个和第三个相同
        assert!(op.cache_hits() >= 1); // 第三个命中缓存
        assert_eq!(op.cache_size(), 2); // "dup" 和 "uniq"
    }

    #[test]
    fn test_batch() {
        let op = FeatureHash::new(1000, 2, "|".into()).unwrap();
        let col_a = vec![Fv::Str("x".into()), Fv::Str("y".into())];
        let col_b = vec![Fv::Str("1".into()), Fv::Str("2".into())];
        let results = op.process_batch(&[&col_a, &col_b], 2).unwrap();
        assert_eq!(results.len(), 2);
        let single_0 = op
            .process(&[Fv::Str("x".into()), Fv::Str("1".into())])
            .unwrap();
        let single_1 = op
            .process(&[Fv::Str("y".into()), Fv::Str("2".into())])
            .unwrap();
        assert_eq!(results[0], single_0);
        assert_eq!(results[1], single_1);
    }

    #[test]
    fn test_mixed_list_and_scalar_single_matches_batch() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let single = op
            .process(&[
                Fv::Str("user".into()),
                Fv::StrList(vec!["a".into(), "b".into()]),
                Fv::Int(7),
            ])
            .unwrap();
        let col_user = vec![Fv::Str("user".into())];
        let col_tags = vec![Fv::StrList(vec!["a".into(), "b".into()])];
        let col_id = vec![Fv::Int(7)];
        let batch = op
            .process_batch(&[&col_user, &col_tags, &col_id], 1)
            .unwrap();
        assert_eq!(batch[0], single);
        assert!(matches!(single, Fv::IntList(ref values) if values.len() == 4));
    }

    #[test]
    fn test_batch_rejects_mixed_scalar_and_list_rows() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let col_a = vec![
            Fv::Str("x".into()),
            Fv::Str("y".into()),
            Fv::StrList(vec!["z".into(), "w".into()]),
        ];
        let col_b = vec![
            Fv::Str("1".into()),
            Fv::Str("2".into()),
            Fv::Str("3".into()),
        ];
        let err = op.process_batch(&[&col_a, &col_b], 3).unwrap_err();
        assert!(err.contains("mixed scalar/list rows"));
    }

    #[test]
    fn test_empty_str_list_input() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let result = op.process(&[Fv::StrList(vec![])]).unwrap();
        assert_eq!(result, Fv::IntList(vec![]));
    }

    #[test]
    fn test_empty_int_list_input() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let result = op.process(&[Fv::IntList(vec![])]).unwrap();
        assert_eq!(result, Fv::IntList(vec![]));
    }

    #[test]
    fn test_non_string_scalar_to_string_fallback() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let i_result = op.process(&[Fv::Int(42)]).unwrap();
        let f_result = op.process(&[Fv::Float(3.14)]).unwrap();
        assert!(matches!(i_result, Fv::Int(v) if v >= 0 && v < 1000));
        assert!(matches!(f_result, Fv::Int(v) if v >= 0 && v < 1000));
        let i42 = op.process(&[Fv::Int(42)]).unwrap();
        let f42_5 = op.process(&[Fv::Float(42.5)]).unwrap();
        assert_ne!(i42, f42_5);
    }

    #[test]
    fn test_single_long_string_doesnt_crash() {
        let op = FeatureHash::new(1000, 1, "|".into()).unwrap();
        let long = "x".repeat(100_000);
        let result = op.process(&[Fv::Str(long)]).unwrap();
        assert!(matches!(result, Fv::Int(v) if v >= 0 && v < 1000));
    }
}
