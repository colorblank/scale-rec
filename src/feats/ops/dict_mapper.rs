//! 字典映射算子：字符串/整数到索引的映射。
use std::collections::HashMap;
use super::{CustomOp, Fv};

pub struct DictMapper {
    mapping: HashMap<String, i32>,
    default_idx: i32,
}

impl DictMapper {
    pub fn new(mapping: HashMap<String, i32>, default_idx: i32) -> Self { Self { mapping, default_idx } }
    pub fn max_idx(&self) -> i32 { self.mapping.values().copied().max().unwrap_or(0) }

    fn map_one(&self, key: &str) -> i32 {
        self.mapping.get(key).copied().unwrap_or(self.default_idx)
    }
}

impl CustomOp for DictMapper {
    fn name(&self) -> &str { "DictMapper" }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        Ok(match &inputs[0] {
            Fv::Str(s) => Fv::Int(self.map_one(s)),
            Fv::Int(i) => Fv::Int(self.map_one(&i.to_string())),
            Fv::StrList(list) => Fv::IntList(list.iter().map(|s| self.map_one(s)).collect()),
            _ => Fv::Int(self.default_idx),
        })
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let out = match &col[i] {
                Fv::Str(s) => Fv::Int(self.map_one(s)),
                Fv::Int(n) => Fv::Int(self.map_one(&n.to_string())),
                Fv::StrList(list) => Fv::IntList(list.iter().map(|s| self.map_one(s)).collect()),
                _ => Fv::Int(self.default_idx),
            };
            results.push(out);
        }
        Ok(results)
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
}
