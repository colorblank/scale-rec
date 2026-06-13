//! 特征配置类型：FlowConfig、SourceDef、OperatorDef、DType、EmbedConfig。

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 列角色：特征入 DAG、训练标签、读入后丢弃。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    Feature,
    Label,
    Discard,
}

/// 数据类型：整数、浮点、字符串、列表。
#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Int,
    Float,
    String,
    Enum {
        values: Vec<String>,
        default: Option<String>,
        oov: Option<String>,
    },
    List {
        dtype: Box<DType>,
        length: usize,
    },
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value =
            serde_yaml::Value::deserialize(deserializer).map_err(serde::de::Error::custom)?;
        dtype_from_yaml_value(&value).map_err(serde::de::Error::custom)
    }
}

impl Serialize for DType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DType::Int => serializer.serialize_str("int"),
            DType::Float => serializer.serialize_str("float"),
            DType::String => serializer.serialize_str("string"),
            DType::Enum {
                values,
                default,
                oov,
            } => {
                let mut outer = serializer.serialize_map(Some(1))?;
                let mut inner = serde_yaml::Mapping::new();
                inner.insert(
                    serde_yaml::Value::String("values".to_string()),
                    serde_yaml::Value::Sequence(
                        values
                            .iter()
                            .cloned()
                            .map(serde_yaml::Value::String)
                            .collect(),
                    ),
                );
                if let Some(default) = default {
                    inner.insert(
                        serde_yaml::Value::String("default".to_string()),
                        serde_yaml::Value::String(default.clone()),
                    );
                }
                if let Some(oov) = oov {
                    inner.insert(
                        serde_yaml::Value::String("oov".to_string()),
                        serde_yaml::Value::String(oov.clone()),
                    );
                }
                outer.serialize_entry("enum", &inner)?;
                outer.end()
            }
            DType::List { dtype, length } => {
                let mut outer = serializer.serialize_map(Some(1))?;
                let mut inner = serde_yaml::Mapping::new();
                inner.insert(
                    serde_yaml::Value::String("item_dtype".to_string()),
                    serde_yaml::to_value(dtype.as_ref()).map_err(serde::ser::Error::custom)?,
                );
                inner.insert(
                    serde_yaml::Value::String("max_len".to_string()),
                    serde_yaml::Value::Number((*length).into()),
                );
                outer.serialize_entry("list", &inner)?;
                outer.end()
            }
        }
    }
}

fn dtype_from_yaml_value(value: &serde_yaml::Value) -> Result<DType, String> {
    match value {
        serde_yaml::Value::String(tag) => match tag.as_str() {
            "int" => Ok(DType::Int),
            "float" => Ok(DType::Float),
            "string" => Ok(DType::String),
            other => Err(format!("unsupported dtype '{}'", other)),
        },
        serde_yaml::Value::Mapping(map) => {
            if let Some(list_spec) = map.get(&serde_yaml::Value::String("list".to_string())) {
                return dtype_list_from_yaml(list_spec);
            }
            if let Some(enum_spec) = map.get(&serde_yaml::Value::String("enum".to_string())) {
                return dtype_enum_from_yaml(enum_spec);
            }
            Err(format!("invalid dtype mapping: {:?}", value))
        }
        _ => Err(format!("invalid dtype: {:?}", value)),
    }
}

fn dtype_list_from_yaml(value: &serde_yaml::Value) -> Result<DType, String> {
    let map = value
        .as_mapping()
        .ok_or_else(|| "list dtype requires mapping".to_string())?;
    let dtype_value = map
        .get(&serde_yaml::Value::String("item_dtype".to_string()))
        .or_else(|| map.get(&serde_yaml::Value::String("dtype".to_string())))
        .ok_or_else(|| "list dtype requires item_dtype".to_string())?;
    let length = map
        .get(&serde_yaml::Value::String("max_len".to_string()))
        .or_else(|| map.get(&serde_yaml::Value::String("length".to_string())))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "list dtype requires max_len".to_string())? as usize;
    Ok(DType::List {
        dtype: Box::new(dtype_from_yaml_value(dtype_value)?),
        length,
    })
}

fn dtype_enum_from_yaml(value: &serde_yaml::Value) -> Result<DType, String> {
    if let Some(seq) = value.as_sequence() {
        let values: Vec<String> = seq.iter().map(yaml_scalar_to_string).collect();
        return Ok(DType::Enum {
            default: values.first().cloned(),
            values,
            oov: None,
        });
    }
    let map = value
        .as_mapping()
        .ok_or_else(|| "enum dtype requires mapping or sequence".to_string())?;
    let values_value = map
        .get(&serde_yaml::Value::String("values".to_string()))
        .ok_or_else(|| "enum dtype requires values".to_string())?;
    let values: Vec<String> = values_value
        .as_sequence()
        .ok_or_else(|| "enum values must be a sequence".to_string())?
        .iter()
        .map(yaml_scalar_to_string)
        .collect();
    if values.is_empty() {
        return Err("enum dtype requires at least one value".to_string());
    }
    let default = map
        .get(&serde_yaml::Value::String("default".to_string()))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| values.first().cloned());
    let oov = map
        .get(&serde_yaml::Value::String("oov".to_string()))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    Ok(DType::Enum {
        values,
        default,
        oov,
    })
}

fn yaml_scalar_to_string(value: &serde_yaml::Value) -> String {
    if let Some(s) = value.as_str() {
        s.to_string()
    } else if let Some(i) = value.as_i64() {
        i.to_string()
    } else if let Some(f) = value.as_f64() {
        f.to_string()
    } else if let Some(b) = value.as_bool() {
        b.to_string()
    } else {
        format!("{:?}", value)
    }
}

/// 变长特征池化策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PoolingStrategy {
    #[default]
    First,
    Flatten,
    Mean,
    Sum,
    Max,
}

/// 序列截断方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TruncationSide {
    /// 保留头部
    #[default]
    Head,
    /// 保留尾部
    Tail,
}

/// 嵌入层配置：词表大小、嵌入维度、池化策略、序列长度、截断方向。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    #[serde(default)]
    pub pooling: PoolingStrategy,
    #[serde(default)]
    pub seq_len: Option<usize>,
    #[serde(default)]
    pub truncation: TruncationSide,
}

/// 原始输入源定义。`embed` 已弃用，全部 embedding 由算子输出配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDef {
    pub name: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub data_source: Option<String>,
    pub dtype: DType,
    pub default_val: String,
    #[serde(default)]
    pub embed: Option<EmbedConfig>, // 已弃用：保留字段兼容旧配置
    #[serde(default)]
    pub role: Role,
    #[serde(default)]
    pub column_index: Option<usize>,
}

/// 算子节点定义。`params` 使用原生 YAML 值，由各算子自行解析。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorDef {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    #[serde(default)]
    pub params: serde_yaml::Value,
    #[serde(default)]
    pub embed: Option<EmbedConfig>,
}

/// 数据源定义：名称、类型及连接参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSourceDef {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub params: serde_yaml::Value,
}

/// 完整的特征编排配置。包含版本、输入源列表和算子列表。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub version: String,
    #[serde(default)]
    pub data_sources: Vec<DataSourceDef>,
    pub sources: Vec<SourceDef>,
    pub operators: Vec<OperatorDef>,
}

impl FlowConfig {
    /// 从 YAML 字符串解析 FlowConfig。
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }
}

/// 严格解析整数，空字符串返回错误。
pub fn parse_int_strict(raw: &str) -> Result<i32, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("empty integer value".to_string());
    }
    text.parse::<i32>()
        .map_err(|e| format!("invalid integer value '{}': {}", raw, e))
}

/// 严格解析浮点数，空字符串返回错误。
pub fn parse_float_strict(raw: &str) -> Result<f32, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("empty float value".to_string());
    }
    text.parse::<f32>()
        .map_err(|e| format!("invalid float value '{}': {}", raw, e))
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn serializes_dtype_with_config_shape() {
        assert_eq!(serde_json::to_value(&DType::Int).unwrap(), "int");
        assert_eq!(serde_json::to_value(&DType::Float).unwrap(), "float");
        assert_eq!(serde_json::to_value(&DType::String).unwrap(), "string");

        let list = DType::List {
            dtype: Box::new(DType::Int),
            length: 3,
        };
        assert_eq!(
            serde_json::to_value(&list).unwrap(),
            serde_json::json!({"list": {"item_dtype": "int", "max_len": 3}})
        );
    }
}
