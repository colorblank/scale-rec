//! 插件算子：通过 cdylib 动态加载外部算子。
use super::{CustomOp, Fv};
use libloading::Library;
use std::any::Any;

/// 外部插件签名: `fn(&[&(dyn Any)]) -> Result<Box<dyn Any>, String>`
type PluginFn =
    unsafe fn(&[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String>;

/// 通过 cdylib 动态加载的外部插件算子。
pub struct PluginOp {
    lib: Library,
    op_name: String,
}

/// 从 YAML params 创建 PluginOp 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let path = params
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or("Missing path for PluginOp")?;
    let op_name = params
        .get("op_name")
        .and_then(|v| v.as_str())
        .unwrap_or("custom_plugin")
        .to_string();
    Ok(Box::new(PluginOp::new(path, op_name)?))
}

impl PluginOp {
    /// 加载动态库并创建插件算子。
    pub fn new(path: &str, op_name: String) -> Result<Self, String> {
        // SAFETY: Loading a dynamic library runs platform loader logic and may execute library
        // initialization code. Callers must only pass trusted plugin paths from controlled config.
        let lib = unsafe {
            Library::new(path).map_err(|e| format!("Failed to load plugin '{}': {}", path, e))?
        };
        Ok(Self { lib, op_name })
    }
}

fn fv_to_any(v: &Fv) -> Box<dyn Any + Send + Sync> {
    match v {
        Fv::Int(i) => Box::new(*i),
        Fv::Float(f) => Box::new(*f),
        Fv::Str(s) => Box::new(s.clone()),
        Fv::IntList(l) => Box::new(l.clone()),
        Fv::FloatList(l) => Box::new(l.clone()),
        Fv::StrList(l) => Box::new(l.clone()),
    }
}

fn any_to_fv(a: Box<dyn Any + Send + Sync>) -> Result<Fv, String> {
    a.downcast::<i32>()
        .map(|v| Fv::Int(*v))
        .or_else(|a| a.downcast::<f32>().map(|v| Fv::Float(*v)))
        .or_else(|a| a.downcast::<String>().map(|v| Fv::Str(*v)))
        .or_else(|a| a.downcast::<Vec<i32>>().map(|v| Fv::IntList(*v)))
        .or_else(|a| a.downcast::<Vec<f32>>().map(|v| Fv::FloatList(*v)))
        .or_else(|a| a.downcast::<Vec<String>>().map(|v| Fv::StrList(*v)))
        .map_err(|_| "PluginOp returned unsupported value type".to_string())
}

impl CustomOp for PluginOp {
    fn name(&self) -> &str {
        &self.op_name
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let any_inputs: Vec<Box<dyn Any + Send + Sync>> = inputs.iter().map(fv_to_any).collect();
        let refs: Vec<&(dyn Any + Send + Sync)> = any_inputs.iter().map(|b| b.as_ref()).collect();
        // SAFETY: The loaded library is kept alive by `self.lib`, and we require the plugin to
        // export `process_custom` with the `PluginFn` ABI and type contract documented above.
        unsafe {
            let func: libloading::Symbol<PluginFn> = self
                .lib
                .get(b"process_custom")
                .map_err(|e| format!("PluginOp: symbol not found: {}", e))?;
            any_to_fv(func(&refs)?)
        }
    }
}
