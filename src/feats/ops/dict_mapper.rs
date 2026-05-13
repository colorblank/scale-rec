//! 字典映射算子：字符串/整数到索引的映射。
use std::any::Any;
use std::collections::HashMap;

/// 字典映射算子。
///
/// 将字符串键映射为整数索引，未命中时返回默认索引。
pub struct DictMapper {
    mapping: HashMap<String, i32>,
    default_idx: i32,
}

impl DictMapper {
    /// 构造映射算子。`default_idx` 为未知键的返回值。
    pub fn new(mapping: HashMap<String, i32>, default_idx: i32) -> Self {
        Self {
            mapping,
            default_idx,
        }
    }
    pub fn max_idx(&self) -> i32 {
        self.mapping.values().copied().max().unwrap_or(0)
    }
}

impl super::CustomOp for DictMapper {
    /// 执行映射: `String` 或 `i32` 输入 → `i32` 索引。
    fn name(&self) -> &str {
        "DictMapper"
    }
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let key: &str = if let Some(s) = inputs[0].downcast_ref::<String>() {
            s.as_str()
        } else if let Some(i) = inputs[0].downcast_ref::<i32>() {
            return Ok(Box::new(
                self.mapping
                    .get(&i.to_string())
                    .copied()
                    .unwrap_or(self.default_idx),
            ));
        } else {
            return Ok(Box::new(self.default_idx));
        };
        Ok(Box::new(
            self.mapping.get(key).copied().unwrap_or(self.default_idx),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
        use crate::feats::ops::CustomOp;

    #[test]
    fn test_known_key() {
        let op = DictMapper::new(
            [("elec".into(), 1), ("book".into(), 2)].into(),
            0,
        );
        let result = op.process(&[&"elec".to_string()]).unwrap();
        assert_eq!(*result.downcast_ref::<i32>().unwrap(), 1);
    }

    #[test]
    fn test_unknown_key_uses_default() {
        let op = DictMapper::new([("a".into(), 1)].into(), 99);
        let result = op.process(&[&"unknown".to_string()]).unwrap();
        assert_eq!(*result.downcast_ref::<i32>().unwrap(), 99);
    }
}
