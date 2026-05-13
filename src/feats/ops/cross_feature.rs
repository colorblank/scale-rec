//! 特征交叉算子：内积或笛卡尔积。
use std::any::Any;

/// 特征交叉算子。
///
/// 支持两种交叉模式：
/// - `"inner_product"` — 两个 `Vec<f32>` 的点积，输出 `f32`
/// - `"cartesian"` — 两个 `Vec<String>` 的笛卡尔积，输出 `Vec<String>`
pub struct CrossFeature {
    cross_type: String,
}

impl CrossFeature {
    /// 构造交叉算子。
    ///
    /// `cross_type` 必须为 `"inner_product"` 或 `"cartesian"`。
    pub fn new(cross_type: String) -> Self {
        Self { cross_type }
    }
}

impl super::CustomOp for CrossFeature {
    fn name(&self) -> &str {
        "CrossFeature"
    }

    /// 执行特征交叉。
    ///
    /// # 输入
    /// - `inner_product` 模式：需要 2 个 `Vec<f32>` 输入，长度必须相等
    /// - `cartesian` 模式：需要 2 个 `Vec<String>` 输入
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        match self.cross_type.as_str() {
            "inner_product" => {
                let a = inputs[0]
                    .downcast_ref::<Vec<f32>>()
                    .ok_or("Expected Vec<f32>")?;
                let b = inputs[1]
                    .downcast_ref::<Vec<f32>>()
                    .ok_or("Expected Vec<f32>")?;
                Ok(Box::new(a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()))
            }
            "cartesian" => {
                let a = inputs[0]
                    .downcast_ref::<Vec<String>>()
                    .ok_or("Expected Vec<String>")?;
                let b = inputs[1]
                    .downcast_ref::<Vec<String>>()
                    .ok_or("Expected Vec<String>")?;
                Ok(Box::new(
                    a.iter()
                        .flat_map(|x| b.iter().map(move |y| format!("{}_{}", x, y)))
                        .collect::<Vec<_>>(),
                ))
            }
            _ => Err(format!("Unknown cross_type: {}", self.cross_type)),
        }
    }
}
