use std::any::Any;
use std::collections::HashMap;

pub struct DictMapper {
    mapping: HashMap<String, i32>,
    default_idx: i32,
}

impl DictMapper {
    pub fn new(mapping: HashMap<String, i32>, default_idx: i32) -> Self {
        Self { mapping, default_idx }
    }
    pub fn max_idx(&self) -> i32 {
        self.mapping.values().copied().max().unwrap_or(0)
    }
}

impl super::CustomOp for DictMapper {
    fn name(&self) -> &str { "DictMapper" }
    fn process(&self, inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String> {
        let key: &str = if let Some(s) = inputs[0].downcast_ref::<String>() { s.as_str() }
            else if let Some(i) = inputs[0].downcast_ref::<i32>() { return Ok(Box::new(self.mapping.get(&i.to_string()).copied().unwrap_or(self.default_idx))); }
            else { return Ok(Box::new(self.default_idx)); };
        Ok(Box::new(self.mapping.get(key).copied().unwrap_or(self.default_idx)))
    }
}
