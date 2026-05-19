//! 表达式算子：Rhai 脚本求值，支持 v0..vN 变量和 log 函数。
use super::{CustomOp, Fv};
use rhai::{Engine, Scope, AST};

pub struct ExpressionOp {
    ast: AST,
    engine: Engine,
}

impl ExpressionOp {
    pub fn new(script: String) -> Self {
        let mut engine = Engine::new();
        engine.register_fn("log", |f: f64| f.ln());
        let ast = engine.compile(&script).expect("Invalid Rhai expression");
        Self { ast, engine }
    }
}

impl CustomOp for ExpressionOp {
    fn name(&self) -> &str {
        "ExpressionOp"
    }
    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let mut scope = Scope::new();
        for (i, v) in inputs.iter().enumerate() {
            let val: f64 = match v {
                Fv::Int(n) => *n as f64,
                Fv::Float(f) => *f as f64,
                _ => 0.0,
            };
            scope.push(format!("v{}", i), val);
        }
        self.engine
            .eval_ast_with_scope::<f64>(&mut scope, &self.ast)
            .map(|r| Fv::Float(r as f32))
            .map_err(|e| e.to_string())
    }
}
