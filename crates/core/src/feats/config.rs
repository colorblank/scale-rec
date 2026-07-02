//! 特征配置类型：FlowConfig、SourceDef、OperatorDef、DType、EmbedConfig。

use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 列角色：特征入 DAG、训练标签、读入后丢弃。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// 普通特征列，会进入 DAG 执行和推理输入契约。
    #[default]
    Feature,
    /// 训练标签列，只在训练侧读取，推理侧不会作为输入暴露。
    Label,
    /// 读入后丢弃的列，用于保留文件兼容但不进入 DAG。
    Discard,
}

/// 数据类型：整数、浮点、字符串、列表。
#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    /// 32 位整数特征。
    Int,
    /// 32 位浮点特征。
    Float,
    /// 字符串特征。
    String,
    /// 有限枚举特征。
    Enum {
        /// 合法枚举值集合。
        values: Vec<String>,
        /// 缺省枚举值。
        default: Option<String>,
        /// 未知枚举值映射目标。
        oov: Option<String>,
    },
    /// 定长列表特征。
    List {
        /// 列表元素类型。
        dtype: Box<DType>,
        /// 列表最大长度或定长长度。
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
    /// 仅使用第一个元素。
    #[default]
    First,
    /// 将定长序列展平成一个向量。
    Flatten,
    /// 对序列 embedding 求平均。
    Mean,
    /// 对序列 embedding 求和。
    Sum,
    /// 对序列 embedding 取逐维最大值。
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
    /// embedding 词表大小。
    pub vocab_size: usize,
    /// embedding 向量维度。
    pub embed_dim: usize,
    /// 序列特征的池化策略。
    #[serde(default)]
    pub pooling: PoolingStrategy,
    /// 序列长度；列表特征可从 schema 推断。
    #[serde(default)]
    pub seq_len: Option<usize>,
    /// 超长序列截断方向。
    #[serde(default)]
    pub truncation: TruncationSide,
}

/// 单个特征的嵌入配置：词表大小、嵌入维度、池化策略等。
#[derive(Debug, Clone)]
pub struct FeatureSpec {
    /// 特征名称。
    pub name: String,
    /// embedding 词表大小。
    pub vocab_size: usize,
    /// embedding 向量维度。
    pub embed_dim: usize,
    /// 序列特征池化策略。
    pub pooling: PoolingStrategy,
    /// 序列长度；标量特征为 `None`。
    pub seq_len: Option<usize>,
    /// 序列截断方向。
    pub truncation: TruncationSide,
}

impl FeatureSpec {
    /// 创建 FeatureSpec，默认使用 First 池化和 Head 截断。
    pub fn new(name: String, vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            name,
            vocab_size,
            embed_dim,
            pooling: PoolingStrategy::First,
            seq_len: None,
            truncation: TruncationSide::Head,
        }
    }

    /// 克隆并修改嵌入维度。
    pub fn with_dim(&self, embed_dim: usize) -> Self {
        Self {
            name: self.name.clone(),
            vocab_size: self.vocab_size,
            embed_dim,
            pooling: self.pooling,
            seq_len: self.seq_len,
            truncation: self.truncation,
        }
    }

    pub fn output_dim(&self) -> usize {
        match (self.pooling, self.seq_len) {
            (PoolingStrategy::Flatten, Some(seq_len)) => self.embed_dim * seq_len,
            _ => self.embed_dim,
        }
    }
}

/// 推荐排序特征的原始来源：用户、物品、上下文或标签。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceKind {
    /// 用户侧输入字段。
    User,
    /// 物品侧输入字段。
    Item,
    /// 请求上下文字段。
    Context,
    /// 训练标签字段。
    Label,
}

/// 原始输入源定义。`embed` 已弃用，全部 embedding 由算子输出配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDef {
    /// 原始字段名称。
    pub name: String,
    /// 字段业务归属。
    #[serde(default)]
    pub source: Option<SourceKind>,
    /// 字段来自的数据源名称。
    #[serde(default)]
    pub data_source: Option<String>,
    /// 原始字段数据类型。
    pub dtype: DType,
    /// 缺省值的字符串表示。
    pub default_val: String,
    /// 已弃用的 source 级 embedding 配置。
    #[serde(default)]
    pub embed: Option<EmbedConfig>, // 已弃用：保留字段兼容旧配置
    /// 字段在训练/推理中的角色。
    #[serde(default)]
    pub role: Role,
    /// 无 header 文件中的列索引。
    #[serde(default)]
    pub column_index: Option<usize>,
}

/// 算子类型枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpType {
    /// 数值分桶算子。
    Bucketing,
    /// 字符串拼接后哈希算子。
    ConcatHash,
    /// 交叉特征算子。
    CrossFeature,
    /// 字典映射算子。
    DictMapper,
    /// 表达式计算算子。
    ExpressionOp,
    /// 特征哈希算子。
    FeatureHash,
    /// 扁平化 split 算子。
    FlatSplit,
    /// JSON 列表抽取算子。
    JsonExtractList,
    /// 列表重叠度算子。
    ListOverlap,
    /// 列表字符串解析算子。
    ListStringParser,
    /// log1p 数值算子。
    Log1p,
    /// 解析与哈希融合算子。
    ParsedFeatureHash,
    /// 动态插件算子。
    PluginOp,
    /// 序列截断/补齐算子。
    SequenceOp,
    /// 字符串 split 算子。
    Split,
    /// 字符串拼接算子。
    StringConcat,
    /// 字符串解析算子。
    StringParser,
    /// 时间解析算子。
    TimeParser,
}

/// 算子节点定义。`params` 使用原生 YAML 值，由各算子自行解析。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorDef {
    /// 算子节点名称。
    pub name: String,
    /// 算子类型。
    pub op_type: OpType,
    /// 输入特征名称列表。
    pub inputs: Vec<String>,
    /// 输出特征名称列表。
    pub outputs: Vec<String>,
    /// 算子参数。
    #[serde(default)]
    pub params: serde_yaml::Value,
    /// 输出 embedding 配置。
    #[serde(default)]
    pub embed: Option<EmbedConfig>,
}

/// 数据源定义：名称、类型及连接参数。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSourceDef {
    /// 数据源名称。
    pub name: String,
    /// 数据源类型。
    pub kind: String,
    /// 数据源说明。
    #[serde(default)]
    pub description: Option<String>,
    /// 数据源连接或读取参数。
    #[serde(default)]
    pub params: serde_yaml::Value,
}

/// 完整的特征编排配置。包含版本、输入源列表和算子列表。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    /// 配置 schema 版本。
    pub version: String,
    /// 外部数据源列表。
    #[serde(default)]
    pub data_sources: Vec<DataSourceDef>,
    /// 原始输入字段列表。
    pub sources: Vec<SourceDef>,
    /// DAG 算子列表。
    pub operators: Vec<OperatorDef>,
}

impl FlowConfig {
    /// 从 YAML 字符串解析 FlowConfig。
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        let mut config: Self = serde_yaml::from_str(yaml)?;
        for source in &mut config.sources {
            if matches!(source.source, Some(SourceKind::Label)) {
                source.role = Role::Label;
            }
        }
        Ok(config)
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
