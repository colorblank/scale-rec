//! 列表重叠检测算子：判断两个列表是否存在共同元素。
use std::any::Any;
use std::collections::HashSet;

/// 列表重叠检测算子。
///
/// 两个 `Vec<String>` 输入，判断是否存在交集。返回 1（有交集）或 0（无交集）。
/// 忽略空字符串元素。
pub struct ListOverlap;

impl ListOverlap {
    pub fn new() -> Self {
        Self
    }
}

impl super::CustomOp for ListOverlap {
    fn name(&self) -> &str {
        "ListOverlap"
    }
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let a: HashSet<&str> = if let Some(list) = inputs[0].downcast_ref::<Vec<String>>() {
            list.iter().filter(|s| !s.is_empty()).map(|s| s.as_str()).collect()
        } else {
            return Ok(Box::new(0i32));
        };
        let b: HashSet<&str> = if let Some(list) = inputs[1].downcast_ref::<Vec<String>>() {
            list.iter().filter(|s| !s.is_empty()).map(|s| s.as_str()).collect()
        } else {
            return Ok(Box::new(0i32));
        };
        if a.is_empty() || b.is_empty() {
            return Ok(Box::new(0i32));
        }
        Ok(Box::new(if a.intersection(&b).next().is_some() { 1i32 } else { 0i32 }))
    }
}
