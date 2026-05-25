//! Static feature schema inference and validation for FlowConfig.

use std::collections::HashMap;

use crate::feats::config::{DType, EmbedConfig, OperatorDef, PoolingStrategy, Role, SourceDef};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureDType {
    Int,
    Float,
    String,
    List {
        dtype: Box<FeatureDType>,
        length: Option<usize>,
    },
    Unknown,
}

impl FeatureDType {
    pub fn is_list(&self) -> bool {
        matches!(self, FeatureDType::List { .. })
    }

    pub fn is_integer_index(&self) -> bool {
        match self {
            FeatureDType::Int => true,
            FeatureDType::List { dtype, .. } => matches!(dtype.as_ref(), FeatureDType::Int),
            _ => false,
        }
    }

    pub fn list_len(&self) -> Option<usize> {
        match self {
            FeatureDType::List { length, .. } => *length,
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSchema {
    pub name: String,
    pub dtype: FeatureDType,
    pub rank: usize,
    pub nullable: bool,
    pub default_val: Option<String>,
    pub cardinality: Option<usize>,
    pub pooling: Option<PoolingStrategy>,
}

pub fn infer_feature_schemas(
    sources: &[SourceDef],
    operators: &[OperatorDef],
) -> Result<HashMap<String, FeatureSchema>, String> {
    let mut schemas = HashMap::new();
    for source in sources {
        if source.role != Role::Feature {
            continue;
        }
        let dtype = dtype_from_config(&source.dtype);
        validate_default(&source.name, &dtype, &source.default_val)?;
        if let Some(embed) = &source.embed {
            validate_embed(&source.name, &dtype, embed)?;
        }
        schemas.insert(
            source.name.clone(),
            FeatureSchema {
                name: source.name.clone(),
                rank: usize::from(dtype.is_list()),
                dtype,
                nullable: false,
                default_val: Some(source.default_val.clone()),
                cardinality: None,
                pooling: None,
            },
        );
    }

    for op in operators {
        let mut inputs = Vec::with_capacity(op.inputs.len());
        for input in &op.inputs {
            inputs.push(
                schemas
                    .get(input)
                    .ok_or_else(|| {
                        format!(
                            "operator '{}' references unknown input '{}'",
                            op.name, input
                        )
                    })?
                    .clone(),
            );
        }
        let inferred = infer_operator_output(op, &inputs)?;
        for output in &op.outputs {
            let mut schema = inferred.clone();
            schema.name = output.clone();
            if let Some(embed) = &op.embed {
                validate_embed(output, &schema.dtype, embed)?;
                schema.cardinality = Some(embed.vocab_size);
                schema.pooling = Some(embed.pooling);
            }
            schemas.insert(output.clone(), schema);
        }
    }

    Ok(schemas)
}

fn dtype_from_config(dtype: &DType) -> FeatureDType {
    match dtype {
        DType::Int => FeatureDType::Int,
        DType::Float => FeatureDType::Float,
        DType::String => FeatureDType::String,
        DType::List { dtype, length } => FeatureDType::List {
            dtype: Box::new(dtype_from_config(dtype)),
            length: Some(*length),
        },
    }
}

fn validate_default(name: &str, dtype: &FeatureDType, default_val: &str) -> Result<(), String> {
    let ok = match dtype {
        FeatureDType::Int => default_val.parse::<f32>().is_ok(),
        FeatureDType::Float => default_val.parse::<f32>().is_ok(),
        FeatureDType::String | FeatureDType::Unknown => true,
        FeatureDType::List { dtype, .. } => validate_default(name, dtype, default_val).is_ok(),
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "source '{}' default '{}' does not match dtype {:?}",
            name, default_val, dtype
        ))
    }
}

fn infer_operator_output(
    op: &OperatorDef,
    inputs: &[FeatureSchema],
) -> Result<FeatureSchema, String> {
    let first = inputs.first();
    let dtype = match op.op_type.as_str() {
        "Bucketing" => {
            require_scalar_number(op, first)?;
            FeatureDType::Int
        }
        "DictMapper" => match first.map(|s| &s.dtype) {
            Some(FeatureDType::List { length, .. }) => FeatureDType::List {
                dtype: Box::new(FeatureDType::Int),
                length: *length,
            },
            _ => FeatureDType::Int,
        },
        "StringParser" | "JsonExtractList" => FeatureDType::List {
            dtype: Box::new(FeatureDType::String),
            length: yaml_usize(&op.params, "pad_len").filter(|v| *v > 0),
        },
        "ListStringParser" => FeatureDType::List {
            dtype: Box::new(FeatureDType::String),
            length: first.and_then(|s| s.dtype.list_len()),
        },
        "Split" | "FlatSplit" => FeatureDType::List {
            dtype: Box::new(FeatureDType::String),
            length: yaml_usize(&op.params, "max_len").filter(|v| *v > 0),
        },
        "CrossFeature" => {
            if yaml_str(&op.params, "cross_type") == Some("inner_product") {
                FeatureDType::Float
            } else {
                FeatureDType::List {
                    dtype: Box::new(FeatureDType::String),
                    length: None,
                }
            }
        }
        "ExpressionOp" => FeatureDType::Float,
        "SequenceOp" => FeatureDType::List {
            dtype: Box::new(FeatureDType::Int),
            length: Some(yaml_usize(&op.params, "max_len").unwrap_or(10)),
        },
        "ListOverlap" => FeatureDType::Int,
        "StringConcat" => FeatureDType::String,
        "FeatureHash" => {
            let has_list_input = inputs.iter().any(|s| s.dtype.is_list());
            let num_hashes = yaml_usize(&op.params, "num_hashes").unwrap_or(1);
            if has_list_input {
                FeatureDType::List {
                    dtype: Box::new(FeatureDType::Int),
                    length: inputs.iter().find_map(|s| s.dtype.list_len()),
                }
            } else if num_hashes > 1 {
                FeatureDType::List {
                    dtype: Box::new(FeatureDType::Int),
                    length: Some(num_hashes),
                }
            } else {
                FeatureDType::Int
            }
        }
        "PluginOp" => FeatureDType::Unknown,
        _ => {
            return Err(format!(
                "Unsupported operator for schema inference: {}",
                op.op_type
            ))
        }
    };
    Ok(FeatureSchema {
        name: op
            .outputs
            .first()
            .cloned()
            .unwrap_or_else(|| op.name.clone()),
        rank: usize::from(dtype.is_list()),
        dtype,
        nullable: false,
        default_val: None,
        cardinality: None,
        pooling: None,
    })
}

fn require_scalar_number(op: &OperatorDef, schema: Option<&FeatureSchema>) -> Result<(), String> {
    match schema.map(|s| &s.dtype) {
        Some(FeatureDType::Int | FeatureDType::Float) => Ok(()),
        Some(dtype) => Err(format!(
            "operator '{}' expects numeric scalar input, got {:?}",
            op.name, dtype
        )),
        None => Err(format!("operator '{}' expects an input", op.name)),
    }
}

fn validate_embed(name: &str, dtype: &FeatureDType, embed: &EmbedConfig) -> Result<(), String> {
    if embed.vocab_size == 0 {
        return Err(format!("embed '{}' vocab_size must be positive", name));
    }
    if embed.embed_dim == 0 {
        return Err(format!("embed '{}' embed_dim must be positive", name));
    }
    if !dtype.is_integer_index() {
        return Err(format!(
            "embeddable feature '{}' must be int or list[int], got {:?}",
            name, dtype
        ));
    }
    if matches!(
        embed.pooling,
        PoolingStrategy::Mean
            | PoolingStrategy::Sum
            | PoolingStrategy::Max
            | PoolingStrategy::Flatten
    ) && !dtype.is_list()
    {
        return Err(format!(
            "embed '{}' pooling {:?} requires list[int]",
            name, embed.pooling
        ));
    }
    if embed.pooling == PoolingStrategy::Flatten
        && embed.seq_len.or_else(|| dtype.list_len()).is_none()
    {
        return Err(format!("embed '{}' pooling flatten requires seq_len", name));
    }
    Ok(())
}

fn yaml_get<'a>(params: &'a serde_yaml::Value, key: &str) -> Option<&'a serde_yaml::Value> {
    params
        .as_mapping()?
        .get(&serde_yaml::Value::String(key.to_string()))
}

fn yaml_str<'a>(params: &'a serde_yaml::Value, key: &str) -> Option<&'a str> {
    yaml_get(params, key)?.as_str()
}

fn yaml_usize(params: &serde_yaml::Value, key: &str) -> Option<usize> {
    yaml_get(params, key)?.as_i64().map(|v| v as usize)
}
