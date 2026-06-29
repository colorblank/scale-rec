//! 序列算子：填充/截断整数序列至固定长度。
use super::{CustomOp, Fv};

/// 整数序列填充/截断至固定长度。
pub struct SequenceOp {
    max_len: usize,
    pad_val: i32,
}
impl SequenceOp {
    /// 创建序列算子。
    pub fn new(max_len: usize, pad_val: i32) -> Self {
        Self { max_len, pad_val }
    }
}

/// 从 YAML params 创建 SequenceOp 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let max_len = params.get("max_len").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let pad_val = params.get("pad_val").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    Ok(Box::new(SequenceOp::new(max_len, pad_val)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_truncate() {
        let op = SequenceOp::new(3, 0);
        let res = op.process(&[Fv::IntList(vec![1, 2, 3, 4, 5])]).unwrap();
        assert_eq!(res, Fv::IntList(vec![1, 2, 3]));
    }

    #[test]
    fn test_sequence_pad() {
        let op = SequenceOp::new(5, -1);
        let res = op.process(&[Fv::IntList(vec![1, 2])]).unwrap();
        assert_eq!(res, Fv::IntList(vec![1, 2, -1, -1, -1]));
    }

    #[test]
    fn test_sequence_exact_length() {
        let op = SequenceOp::new(3, -1);
        let res = op.process(&[Fv::IntList(vec![1, 2, 3])]).unwrap();
        assert_eq!(res, Fv::IntList(vec![1, 2, 3]));
    }

    #[test]
    fn test_sequence_empty_input() {
        let op = SequenceOp::new(3, 0);
        let res = op.process(&[Fv::IntList(vec![])]).unwrap();
        assert_eq!(res, Fv::IntList(vec![0, 0, 0]));
    }

    #[test]
    fn test_sequence_max_len_zero() {
        let op = SequenceOp::new(0, 0);
        let res = op.process(&[Fv::IntList(vec![1, 2, 3])]).unwrap();
        assert_eq!(res, Fv::IntList(vec![]));
    }

    #[test]
    fn test_sequence_wrong_type() {
        let op = SequenceOp::new(3, 0);
        let err = op.process(&[Fv::Str("hello".into())]).unwrap_err();
        assert!(err.contains("Expected IntList"));
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
