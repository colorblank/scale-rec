//! 连续值分桶算子：将浮点数映射为桶索引。
use super::{CustomOp, Fv};

pub struct Bucketing {
    boundaries: Vec<f32>,
}

impl Bucketing {
    pub fn new(mut boundaries: Vec<f32>) -> Self {
        boundaries.sort_by(|a, b| a.total_cmp(b));
        Self { boundaries }
    }
}

impl CustomOp for Bucketing {
    fn name(&self) -> &str { "Bucketing" }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let val: f32 = match &inputs[0] {
            Fv::Float(f) => *f,
            Fv::Int(i) => *i as f32,
            _ => return Err("Bucketing: expected float".into()),
        };
        let mut bucket = 0i32;
        for b in &self.boundaries {
            if val >= *b { bucket += 1; } else { break; }
        }
        Ok(Fv::Int(bucket))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let v: f32 = match &col[i] {
                Fv::Float(f) => *f,
                Fv::Int(n) => *n as f32,
                _ => return Err("Bucketing batch: expected float".into()),
            };
            let mut bucket = 0i32;
            for b in &self.boundaries {
                if v >= *b { bucket += 1; } else { break; }
            }
            results.push(Fv::Int(bucket));
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feats::ops::CustomOp;

    #[test]
    fn test_bucketing() {
        let op = Bucketing::new(vec![18.0, 25.0, 35.0, 50.0]);
        let r = op.process(&[Fv::Float(28.5)]).unwrap();
        assert_eq!(r, Fv::Int(2));
    }
    #[test]
    fn test_bucketing_below() {
        let op = Bucketing::new(vec![10.0]);
        assert_eq!(op.process(&[Fv::Float(5.0)]).unwrap(), Fv::Int(0));
    }
    #[test]
    fn test_bucketing_above() {
        let op = Bucketing::new(vec![10.0, 20.0]);
        assert_eq!(op.process(&[Fv::Float(30.0)]).unwrap(), Fv::Int(2));
    }
}
