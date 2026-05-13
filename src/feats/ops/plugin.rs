use libloading::{Library, Symbol};
use std::any::Any;

pub struct PluginOp {
    lib: Library,
    op_name: String,
}

impl PluginOp {
    pub fn new(path: &str, op_name: String) -> Result<Self, String> {
        let lib = unsafe { Library::new(path).map_err(|e| format!("Failed to load plugin: {}", e))? };
        Ok(Self { lib, op_name })
    }
}

impl super::CustomOp for PluginOp {
    fn name(&self) -> &str { &self.op_name }
    fn process(&self, _inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String> {
        Err("PluginOp not fully implemented".into())
    }
}
