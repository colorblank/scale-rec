use std::any::Any;

pub struct CrossFeature { cross_type: String }

impl CrossFeature {
    pub fn new(cross_type: String) -> Self { Self { cross_type } }
}

impl super::CustomOp for CrossFeature {
    fn name(&self) -> &str { "CrossFeature" }
    fn process(&self, inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String> {
        match self.cross_type.as_str() {
            "inner_product" => {
                let a = inputs[0].downcast_ref::<Vec<f32>>().ok_or("Expected Vec<f32>")?;
                let b = inputs[1].downcast_ref::<Vec<f32>>().ok_or("Expected Vec<f32>")?;
                Ok(Box::new(a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()))
            }
            "cartesian" => {
                let a = inputs[0].downcast_ref::<Vec<String>>().ok_or("Expected Vec<String>")?;
                let b = inputs[1].downcast_ref::<Vec<String>>().ok_or("Expected Vec<String>")?;
                Ok(Box::new(a.iter().flat_map(|x| b.iter().map(move |y| format!("{}_{}", x, y))).collect::<Vec<_>>()))
            }
            _ => Err(format!("Unknown cross_type: {}", self.cross_type)),
        }
    }
}
