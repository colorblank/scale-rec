//! 插件算子：通过 cdylib 动态加载外部算子。
use libloading::Library;
use std::any::Any;

/// 外部插件签名: `fn(&[FeatureValue]) -> Result<Box<dyn Any>, String>`
type PluginFn = unsafe fn(
    inputs: &[&(dyn Any + Send + Sync)],
) -> Result<Box<dyn Any + Send + Sync>, String>;

pub struct PluginOp {
    lib: Library,
    op_name: String,
}

impl PluginOp {
    pub fn new(path: &str, op_name: String) -> Result<Self, String> {
        let lib =
            unsafe { Library::new(path).map_err(|e| format!("Failed to load plugin: {}", e))? };
        Ok(Self { lib, op_name })
    }
}

impl super::CustomOp for PluginOp {
    fn name(&self) -> &str {
        &self.op_name
    }

    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        unsafe {
            let func: libloading::Symbol<PluginFn> = self
                .lib
                .get(b"process_custom")
                .map_err(|e| format!("PluginOp: symbol 'process_custom' not found: {}", e))?;
            func(inputs)
        }
    }
}
