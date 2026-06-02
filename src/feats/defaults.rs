//! Shared default value parsing for Rust feature execution paths.

use crate::feats::config::{parse_float_strict, parse_int_strict, DType, SourceDef};
use crate::feats::ops::Fv;

pub fn parse_source_default(source: &SourceDef) -> Fv {
    parse_default(&source.default_val, &source.dtype)
}

pub fn parse_default(raw: &str, dtype: &DType) -> Fv {
    match dtype {
        DType::Int => Fv::Int(parse_int_strict(raw).unwrap_or(0)),
        DType::Float => Fv::Float(parse_float_strict(raw).unwrap_or(0.0)),
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
            DType::Int => Fv::IntList(vec![parse_int_strict(raw).unwrap_or(0); *length]),
            DType::Float => Fv::FloatList(vec![parse_float_strict(raw).unwrap_or(0.0); *length]),
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
            _ => Fv::Int(0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_default_uses_oov_for_unknown_value() {
        let dtype = DType::Enum {
            values: vec!["known".to_string()],
            default: None,
            oov: Some("other".to_string()),
        };

        let parsed = parse_default("missing", &dtype);

        assert_eq!(parsed, Fv::Str("other".to_string()));
    }

    #[test]
    fn list_default_repeats_to_configured_length() {
        let dtype = DType::List {
            dtype: Box::new(DType::Int),
            length: 3,
        };

        let parsed = parse_default("7", &dtype);

        assert_eq!(parsed, Fv::IntList(vec![7, 7, 7]));
    }
}
