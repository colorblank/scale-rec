use rhai::{Engine, Scope};
use std::any::Any;

pub struct ExpressionOp { script: String, engine: Engine }

impl ExpressionOp {
    pub fn new(script: String) -> Self {
        let mut engine = Engine::new();
        engine.register_fn("log", |f: f64| f.ln());
        Self { script, engine }
    }
}

impl super::CustomOp for ExpressionOp {
    fn name(&self) -> &str { "ExpressionOp" }
    fn process(&self, inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String> {
        let mut scope = Scope::new();
        for (i, val) in inputs.iter().enumerate() {
            let v: f64 = if let Some(f) = val.downcast_ref::<f32>() { *f as f64 }
                else if let Some(f) = val.downcast_ref::<f64>() { *f }
                else if let Some(i) = val.downcast_ref::<i32>() { *i as f64 }
                else { 0.0 };
            scope.push(format!("v{}", i), v);
        }
        self.engine.eval_ast_with_scope::<f64>(&mut scope, &self.engine.compile(&self.script).map_err(|e| e.to_string())?).map(|v| Box::new(v as f32) as Box<dyn Any + Send + Sync>).map_err(|e| e.to_string())
    }
}
