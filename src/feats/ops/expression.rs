//! 表达式算子：Rhai 脚本求值，支持 v0..vN 变量和 log 函数。
use rhai::{Engine, Scope, AST};
use std::any::Any;

/// Rhai 脚本求值算子。
///
/// 将输入绑定为 v0..vN 变量，执行自定义表达式并返回 f32。
/// 内置 `log`(ln) 函数。AST 在构造时编译一次，后续调用复用。
pub struct ExpressionOp {
    ast: AST,
    engine: Engine,
}

impl ExpressionOp {
    /// 构造表达式算子，`script` 为 Rhai 表达式字符串。
    /// 编译失败时 panic——脚本语法错误应在配置加载阶段被捕获。
    pub fn new(script: String) -> Self {
        let mut engine = Engine::new();
        engine.register_fn("log", |f: f64| f.ln());
        let ast = engine
            .compile(&script)
            .expect("Invalid Rhai expression script");
        Self { ast, engine }
    }
}

impl super::CustomOp for ExpressionOp {
    fn name(&self) -> &str {
        "ExpressionOp"
    }
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String> {
        let mut scope = Scope::new();
        for (i, val) in inputs.iter().enumerate() {
            let v: f64 = if let Some(f) = val.downcast_ref::<f32>() {
                *f as f64
            } else if let Some(f) = val.downcast_ref::<f64>() {
                *f
            } else if let Some(i) = val.downcast_ref::<i32>() {
                *i as f64
            } else {
                0.0
            };
            scope.push(format!("v{}", i), v);
        }
        self.engine
            .eval_ast_with_scope::<f64>(&mut scope, &self.ast)
            .map(|v| Box::new(v as f32) as Box<dyn Any + Send + Sync>)
            .map_err(|e| e.to_string())
    }
}
