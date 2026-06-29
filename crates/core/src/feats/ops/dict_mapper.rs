//! 字典映射算子：字符串/整数到索引的映射。
use super::{CustomOp, Fv, OpExecutionStats};
use std::collections::HashMap;

/// 字典映射算子。
///
/// 约定：mapping 值从 1 起始，0 保留为 default_idx 表示未命中/缺失。
/// 这样下游 Embedding 的 index 0 可固定映射为零向量，避免 padding 与真实特征混淆。
pub struct DictMapper {
    mapping: HashMap<String, i32>,
    default_idx: i32,
}

impl DictMapper {
    /// 创建字典映射算子。
    pub fn new(mapping: HashMap<String, i32>, default_idx: i32) -> Self {
        Self {
            mapping,
            default_idx,
        }
    }
    /// 返回 mapping 中最大的索引值。
    pub fn max_idx(&self) -> i32 {
        self.mapping.values().copied().max().unwrap_or(0)
    }

    fn map_one(&self, key: &str) -> i32 {
        self.mapping.get(key).copied().unwrap_or(self.default_idx)
    }

    fn map_one_with_hit(&self, key: &str) -> (i32, bool) {
        match self.mapping.get(key) {
            Some(value) => (*value, false),
            None => (self.default_idx, true),
        }
    }
}

/// 从 YAML params 创建 DictMapper 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let default_idx = params
        .get("default_idx")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;
    let mapping = params
        .get("mapping")
        .and_then(|v| v.as_mapping())
        .map(|map| {
            map.iter()
                .filter_map(|(k, v)| Some((k.as_str()?.to_string(), v.as_i64()? as i32)))
                .collect::<HashMap<String, i32>>()
        })
        .unwrap_or_default();
    Ok(Box::new(DictMapper::new(mapping, default_idx)))
}

impl CustomOp for DictMapper {
    fn name(&self) -> &str {
        "DictMapper"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        Ok(match &inputs[0] {
            Fv::Str(s) => Fv::Int(self.map_one(s)),
            Fv::Int(i) => Fv::Int(self.map_one(&i.to_string())),
            Fv::StrList(list) => Fv::IntList(list.iter().map(|s| self.map_one(s)).collect()),
            _ => Fv::Int(self.default_idx),
        })
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        self.process_batch_with_stats(inputs, n_rows)
            .map(|(output, _)| output)
    }

    fn process_batch_with_stats(
        &self,
        inputs: &[&[Fv]],
        n_rows: usize,
    ) -> Result<(Vec<Fv>, OpExecutionStats), String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        let mut default_hits = 0u64;
        for i in 0..n_rows {
            let out = match &col[i] {
                Fv::Str(s) => {
                    let (value, missed) = self.map_one_with_hit(s);
                    default_hits += u64::from(missed);
                    Fv::Int(value)
                }
                Fv::Int(n) => {
                    let (value, missed) = self.map_one_with_hit(&n.to_string());
                    default_hits += u64::from(missed);
                    Fv::Int(value)
                }
                Fv::StrList(list) => Fv::IntList(
                    list.iter()
                        .map(|key| {
                            let (value, missed) = self.map_one_with_hit(key);
                            default_hits += u64::from(missed);
                            value
                        })
                        .collect(),
                ),
                _ => {
                    default_hits += 1;
                    Fv::Int(self.default_idx)
                }
            };
            results.push(out);
        }
        Ok((
            results,
            OpExecutionStats {
                dict_mapper_default_hits: default_hits,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feats::ops::CustomOp;

    #[test]
    fn test_known_key() {
        let op = DictMapper::new([("elec".into(), 1), ("book".into(), 2)].into(), 0);
        assert_eq!(op.process(&[Fv::Str("elec".into())]).unwrap(), Fv::Int(1));
    }
    #[test]
    fn test_unknown_default() {
        let op = DictMapper::new([("a".into(), 1)].into(), 99);
        assert_eq!(op.process(&[Fv::Str("x".into())]).unwrap(), Fv::Int(99));
    }

    #[test]
    fn empty_map_returns_default() {
        let op = DictMapper::new(HashMap::new(), 0);
        assert_eq!(
            op.process(&[Fv::Str("anything".into())]).unwrap(),
            Fv::Int(0)
        );
    }

    #[test]
    fn float_input_falls_back_to_default() {
        let op = DictMapper::new([("a".into(), 1)].into(), 0);
        let result = op.process(&[Fv::Float(99.9)]).unwrap();
        assert_eq!(result, Fv::Int(0));
    }

    #[test]
    fn int_input_matches_string_key() {
        let mut m = HashMap::new();
        m.insert("42".into(), 7);
        let op = DictMapper::new(m, 0);
        assert_eq!(op.process(&[Fv::Int(42)]).unwrap(), Fv::Int(7));
        assert_eq!(op.process(&[Fv::Int(99)]).unwrap(), Fv::Int(0));
    }

    #[test]
    fn empty_str_list_returns_empty_list() {
        let op = DictMapper::new([("a".into(), 1)].into(), 0);
        let result = op.process(&[Fv::StrList(vec![])]).unwrap();
        assert_eq!(result, Fv::IntList(vec![]));
    }

    #[test]
    fn batch_stats_count_only_actual_mapping_misses() {
        let op = DictMapper::new(
            [("known".into(), 1), ("same_as_default".into(), 99)].into(),
            99,
        );
        let input = vec![
            Fv::Str("known".into()),
            Fv::Str("missing".into()),
            Fv::Str("same_as_default".into()),
            Fv::StrList(vec!["known".into(), "missing2".into()]),
            Fv::Float(1.0),
        ];

        let (_, stats) = op.process_batch_with_stats(&[&input], input.len()).unwrap();

        assert_eq!(stats.dict_mapper_default_hits, 3);
    }
}
