//! 特征交叉算子：内积或笛卡尔积。
use super::{CustomOp, Fv};

/// 特征交叉算子：内积或笛卡尔积。
pub struct CrossFeature {
    cross_type: String,
}
impl CrossFeature {
    /// 创建交叉特征算子。
    pub fn new(cross_type: String) -> Self {
        Self { cross_type }
    }
}

pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let cross_type = params.get("cross_type").and_then(|v| v.as_str()).unwrap_or("cartesian").to_string();
    Ok(Box::new(CrossFeature::new(cross_type)))
}

impl CustomOp for CrossFeature {
    fn name(&self) -> &str {
        "CrossFeature"
    }
    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        if inputs.len() != 2 {
            return Err(format!(
                "CrossFeature expects exactly 2 inputs, got {}",
                inputs.len()
            ));
        }
        match self.cross_type.as_str() {
            "inner_product" => {
                let a: Vec<f32> = match &inputs[0] {
                    Fv::IntList(v) => v.iter().map(|&x| x as f32).collect(),
                    _ => return Err("Expected IntList".into()),
                };
                let b: Vec<f32> = match &inputs[1] {
                    Fv::IntList(v) => v.iter().map(|&x| x as f32).collect(),
                    _ => return Err("Expected IntList".into()),
                };
                Ok(Fv::Float(a.iter().zip(&b).map(|(x, y)| x * y).sum()))
            }
            _ => {
                let a = match &inputs[0] {
                    Fv::StrList(v) => v,
                    _ => return Err("Expected StrList".into()),
                };
                let b = match &inputs[1] {
                    Fv::StrList(v) => v,
                    _ => return Err("Expected StrList".into()),
                };
                let mut r = Vec::new();
                for x in a {
                    for y in b {
                        r.push(format!("{}_{}", x, y));
                    }
                }
                Ok(Fv::StrList(r))
            }
        }
    }
}
