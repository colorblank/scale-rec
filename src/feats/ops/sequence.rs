//! 序列算子：填充/截断整数序列至固定长度。
use super::{CustomOp, Fv};

pub struct SequenceOp {
    max_len: usize,
    pad_val: i32,
}
impl SequenceOp {
    pub fn new(max_len: usize, pad_val: i32) -> Self {
        Self { max_len, pad_val }
    }
}

impl CustomOp for SequenceOp {
    fn name(&self) -> &str {
        "SequenceOp"
    }
    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let seq = match &inputs[0] {
            Fv::IntList(l) => l,
            _ => return Err("Expected IntList".into()),
        };
        let mut result = seq.clone();
        result.resize(self.max_len, self.pad_val);
        Ok(Fv::IntList(result))
    }
}
