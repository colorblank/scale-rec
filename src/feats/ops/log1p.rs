//! log1p 数值算子：计算 ln(1 + x)。
use super::{CustomOp, Fv};

/// 单输入 log1p 算子。
pub struct Log1p;

impl Log1p {
    /// 创建 log1p 算子。
    pub fn new() -> Self {
        Self
    }

    fn value(input: &Fv) -> Result<f32, String> {
        let value = match input {
            Fv::Int(n) => *n as f32,
            Fv::Float(f) => *f,
            _ => return Err("Log1p: expected numeric scalar".into()),
        };
        if value <= -1.0 {
            return Err("Log1p: input must be greater than -1".into());
        }
        Ok(value)
    }
}

impl Default for Log1p {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomOp for Log1p {
    fn name(&self) -> &str {
        "Log1p"
    }

    fn process(&self, inputs: &[Fv]) -> Result<Fv, String> {
        let value = Self::value(&inputs[0])?;
        Ok(Fv::Float(value.ln_1p()))
    }

    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let col = inputs[0];
        let mut results = Vec::with_capacity(n_rows);
        for input in col.iter().take(n_rows) {
            let value = Self::value(input)?;
            results.push(Fv::Float(value.ln_1p()));
        }
        Ok(results)
    }
}

/// 从 YAML params 创建 Log1p 算子。
pub fn create(_params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    Ok(Box::new(Log1p::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feats::ops::CustomOp;

    #[test]
    fn test_log1p_float() {
        let op = Log1p::new();
        let result = op.process(&[Fv::Float(5999.0)]).unwrap();
        match result {
            Fv::Float(value) => assert!((value - 8.699_515).abs() < 1e-6),
            other => panic!("expected float, got {:?}", other),
        }
    }

    #[test]
    fn test_log1p_int() {
        let op = Log1p::new();
        assert_eq!(op.process(&[Fv::Int(0)]).unwrap(), Fv::Float(0.0));
    }

    #[test]
    fn test_log1p_rejects_domain_error() {
        let op = Log1p::new();
        assert!(op.process(&[Fv::Float(-1.0)]).is_err());
    }
}
