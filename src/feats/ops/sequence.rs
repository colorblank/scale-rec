use std::any::Any;

pub struct SequenceOp { max_len: usize, pad_val: i32 }

impl SequenceOp {
    pub fn new(max_len: usize, pad_val: i32) -> Self { Self { max_len, pad_val } }
}

impl super::CustomOp for SequenceOp {
    fn name(&self) -> &str { "SequenceOp" }
    fn process(&self, inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String> {
        let seq = inputs[0].downcast_ref::<Vec<i32>>().ok_or("Expected Vec<i32>")?;
        let mut result = seq.clone();
        while result.len() < self.max_len { result.push(self.pad_val); }
        result.truncate(self.max_len);
        Ok(Box::new(result))
    }
}
