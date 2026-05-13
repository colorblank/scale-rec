//! 连续值分桶算子：将浮点数映射为桶索引。
use std::any::Any;

pub struct Bucketing {
    boundaries: Vec<f32>,
}

impl Bucketing {
    pub fn new(mut boundaries: Vec<f32>) -> Self {
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { boundaries }
    }
}

impl super::CustomOp for Bucketing {
    fn name(&self) -> &str {
        "Bucketing"
    }
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let val: f32 = if let Some(f) = inputs[0].downcast_ref::<f32>() {
            *f
        } else if let Some(f) = inputs[0].downcast_ref::<f64>() {
            *f as f32
        } else {
            return Err("Bucketing: expected f32/f64".into());
        };
        let mut bucket = 0i32;
        for b in &self.boundaries {
            if val >= *b {
                bucket += 1;
            } else {
                break;
            }
        }
        Ok(Box::new(bucket))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
        use crate::feats::ops::CustomOp;

    #[test]
    fn test_bucketing() {
        let op = Bucketing::new(vec![18.0, 25.0, 35.0, 50.0]);
        let result = op.process(&[&28.5f32]).unwrap();
        let bucket = result.downcast_ref::<i32>().unwrap();
        assert_eq!(*bucket, 2); // 25 <= 28.5 < 35
    }

    #[test]
    fn test_bucketing_below_range() {
        let op = Bucketing::new(vec![10.0]);
        let result = op.process(&[&5.0f32]).unwrap();
        assert_eq!(*result.downcast_ref::<i32>().unwrap(), 0);
    }

    #[test]
    fn test_bucketing_above_range() {
        let op = Bucketing::new(vec![10.0, 20.0]);
        let result = op.process(&[&30.0f64]).unwrap();
        assert_eq!(*result.downcast_ref::<i32>().unwrap(), 2);
    }
}
