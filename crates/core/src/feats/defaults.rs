//! Shared default value parsing for Rust feature execution paths.

use crate::feats::config::{parse_float_strict, parse_int_strict, DType, SourceDef};
use crate::feats::ops::Fv;
use tracing::warn;

/// 根据 SourceDef 的 dtype 和 default_val 生成 Fv 默认值。
pub fn source_default(source: &SourceDef) -> Fv {
    parse_default(&source.default_val, &source.dtype)
}

fn parse_default(raw: &str, dtype: &DType) -> Fv {
    match dtype {
        DType::Int => match parse_int_strict(raw) {
            Ok(value) => Fv::Int(value),
            Err(err) => {
                warn!(raw = %raw, dtype = "int", error = %err, "invalid default value, falling back to 0");
                Fv::Int(0)
            }
        },
        DType::Float => match parse_float_strict(raw) {
            Ok(value) => Fv::Float(value),
            Err(err) => {
                warn!(raw = %raw, dtype = "float", error = %err, "invalid default value, falling back to 0.0");
                Fv::Float(0.0)
            }
        },
        DType::String => Fv::Str(raw.to_string()),
        DType::Enum {
            values,
            default,
            oov,
        } => {
            let value = default.as_deref().unwrap_or(raw);
            Fv::Str(normalize_enum_default(value, raw, values, oov))
        }
        DType::List { dtype, length } => match dtype.as_ref() {
            DType::Int => match parse_int_strict(raw) {
                Ok(value) => Fv::IntList(vec![value; *length]),
                Err(err) => {
                    warn!(raw = %raw, dtype = "list[int]", error = %err, "invalid default value, falling back to 0");
                    Fv::IntList(vec![0; *length])
                }
            },
            DType::Float => match parse_float_strict(raw) {
                Ok(value) => Fv::FloatList(vec![value; *length]),
                Err(err) => {
                    warn!(raw = %raw, dtype = "list[float]", error = %err, "invalid default value, falling back to 0.0");
                    Fv::FloatList(vec![0.0; *length])
                }
            },
            DType::String => Fv::StrList(vec![raw.to_string(); *length]),
            DType::Enum {
                values,
                default,
                oov,
            } => {
                let value = default.as_deref().unwrap_or(raw);
                Fv::StrList(vec![
                    normalize_enum_default(value, raw, values, oov);
                    *length
                ])
            }
            _ => {
                warn!(raw = %raw, dtype = "list<unknown>", "unsupported list default dtype, falling back to 0");
                Fv::Int(0)
            }
        },
    }
}

fn normalize_enum_default(
    value: &str,
    raw: &str,
    values: &[String],
    oov: &Option<String>,
) -> String {
    if values.iter().any(|candidate| candidate == value) {
        value.to_string()
    } else {
        oov.as_deref().unwrap_or(raw).to_string()
    }
}

/// 将训练侧 pandas 字符串值按 dtype 解析为 Fv。
/// 严格按 dtype 解析；若解析失败（如 int 列收到非数字字符串）则返回 Err。
/// 调用方应处理 Err（例如回退到 default_val）。
pub fn parse_string_to_fv(raw: &str, dtype: &DType) -> Result<Fv, String> {
    match dtype {
        DType::Int => raw
            .parse::<i32>()
            .map(Fv::Int)
            .map_err(|e| format!("parse int from '{}': {}", raw, e)),
        DType::Float => raw
            .parse::<f32>()
            .map(Fv::Float)
            .map_err(|e| format!("parse float from '{}': {}", raw, e)),
        DType::String => Ok(Fv::Str(raw.to_string())),
        DType::Enum { .. } => Ok(Fv::Str(raw.to_string())),
        DType::List { .. } => Ok(Fv::Str(raw.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feats::config::SourceDef;

    #[test]
    fn enum_default_uses_oov_for_unknown_value() {
        let source = SourceDef {
            name: "feature".to_string(),
            source: None,
            data_source: None,
            dtype: DType::Enum {
                values: vec!["known".to_string()],
                default: None,
                oov: Some("other".to_string()),
            },
            default_val: "missing".to_string(),
            embed: None,
            role: crate::feats::config::Role::Feature,
            column_index: None,
        };

        assert_eq!(source_default(&source), Fv::Str("other".to_string()));
    }

    #[test]
    fn list_default_repeats_to_configured_length() {
        let source = SourceDef {
            name: "feature".to_string(),
            source: None,
            data_source: None,
            dtype: DType::List {
                dtype: Box::new(DType::Int),
                length: 3,
            },
            default_val: "7".to_string(),
            embed: None,
            role: crate::feats::config::Role::Feature,
            column_index: None,
        };

        assert_eq!(source_default(&source), Fv::IntList(vec![7, 7, 7]));
    }
}
