//! 表达式算子：Rhai 脚本求值，支持 v0..vN 变量和 log 函数。
use super::{CustomOp, Fv};
use rhai::{Engine, Scope, AST};
use tracing::warn;

/// Rhai 脚本表达式求值算子。
pub struct ExpressionOp {
    ast: AST,
    engine: Engine,
}

impl ExpressionOp {
    /// 创建表达式算子，预编译 Rhai 脚本。
    pub fn new(script: String) -> Result<Self, String> {
        let mut engine = Engine::new();
        engine.register_fn("log", |f: f64| f.ln());
        let ast = engine
            .compile(&script)
            .map_err(|e| format!("invalid Rhai expression '{}': {}", script, e))?;
        Ok(Self { ast, engine })
    }
}

/// 从 YAML params 创建 ExpressionOp 算子。
pub fn create(params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    let script = params
        .get("script")
        .and_then(|v| v.as_str())
        .ok_or("Missing script for ExpressionOp")?
        .to_string();
    Ok(Box::new(ExpressionOp::new(script)?))
}

impl CustomOp for ExpressionOp {
    fn name(&self) -> &str {
        "ExpressionOp"
    }
    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let mut scope = Scope::new();
        for (i, v) in inputs.iter().enumerate() {
            let val = numeric_input(v, i);
            scope.push(format!("v{}", i), val);
        }
        self.engine
            .eval_ast_with_scope::<f64>(&mut scope, &self.ast)
            .map(|r| Fv::Float(r as f32))
            .map_err(|e| e.to_string())
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let variable_names: Vec<String> = (0..inputs.len()).map(|i| format!("v{}", i)).collect();
        let mut scope = Scope::new();
        for name in &variable_names {
            scope.push(name.as_str(), 0.0_f64);
        }

        let mut results = Vec::with_capacity(n_rows);
        for row in 0..n_rows {
            for (i, col) in inputs.iter().enumerate() {
                scope.set_value(variable_names[i].as_str(), numeric_input(&col[row], i));
            }
            let value = self
                .engine
                .eval_ast_with_scope::<f64>(&mut scope, &self.ast)
                .map_err(|e| e.to_string())?;
            results.push(Fv::Float(value as f32));
        }
        Ok(results)
    }
}

fn numeric_input(value: &Fv, input_idx: usize) -> f64 {
    match value {
        Fv::Int(n) => *n as f64,
        Fv::Float(f) => *f as f64,
        other => {
            warn!(
                input_idx,
                input_type = %other.type_name(),
                "non-numeric expression input, falling back to 0.0"
            );
            0.0
        }
    }
}
