//! 序列算子：填充/截断整数序列至固定长度。
use std::any::Any;

/// 序列填充/截断算子。
///
/// 将 `Vec<i32>` 填充或截断至固定长度 `max_len`，使用 `pad_val` 填充。
pub struct SequenceOp {
    max_len: usize,
    pad_val: i32,
}

impl SequenceOp {
    /// 构造序列算子，`max_len` 为目标长度，`pad_val` 为填充值。
    pub fn new(max_len: usize, pad_val: i32) -> Self {
        Self { max_len, pad_val }
    }
}

impl super::CustomOp for SequenceOp {
    /// 执行填充/截断: `Vec<i32>` → `Vec<i32>`(长度 = max_len)。
    fn name(&self) -> &str {
        "SequenceOp"
    }
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let seq = inputs[0]
            .downcast_ref::<Vec<i32>>()
            .ok_or("Expected Vec<i32>")?;
        let mut result = seq.clone();
        while result.len() < self.max_len {
            result.push(self.pad_val);
        }
        result.truncate(self.max_len);
        Ok(Box::new(result))
    }
}
