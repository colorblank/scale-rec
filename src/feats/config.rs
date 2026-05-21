//! 特征配置类型：FlowConfig、SourceDef、OperatorDef、DType、EmbedConfig。

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Int,
    Float,
    String,
    List { dtype: Box<DType>, length: usize },
}

/// 变长特征池化策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PoolingStrategy {
    #[default]
    Flatten,
    Mean,
    Sum,
    Max,
}

/// 嵌入层配置：词表大小、嵌入维度、池化策略。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    #[serde(default)]
    pub pooling: PoolingStrategy,
}

/// 原始输入源定义。`embed` 已弃用，全部 embedding 由算子输出配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDef {
    pub name: String,
    #[serde(default)]
    pub source: Option<String>,
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

/// 完整的特征编排配置。包含版本、输入源列表和算子列表。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    pub version: String,
    pub sources: Vec<SourceDef>,
    pub operators: Vec<OperatorDef>,
}

impl FlowConfig {
    pub fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(yaml)
    }
}
