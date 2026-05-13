use std::any::Any;

pub struct Bucketing { boundaries: Vec<f32> }

impl Bucketing {
    pub fn new(mut boundaries: Vec<f32>) -> Self {
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { boundaries }
    }
}

impl super::CustomOp for Bucketing {
    fn name(&self) -> &str { "Bucketing" }
    fn process(&self, inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String> {
        let val: f32 = if let Some(f) = inputs[0].downcast_ref::<f32>() { *f }
            else if let Some(f) = inputs[0].downcast_ref::<f64>() { *f as f32 }
            else { return Err("Bucketing: expected f32/f64".into()); };
        let mut bucket = 0i32;
        for b in &self.boundaries { if val >= *b { bucket += 1; } else { break; } }
        Ok(Box::new(bucket))
    }
}
