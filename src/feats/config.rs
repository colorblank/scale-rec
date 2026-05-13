//! 特征配置类型：FlowConfig、SourceDef、OperatorDef、DType、EmbedConfig。

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Int,
    Float,
    String,
    List { dtype: Box<DType>, length: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDef {
    pub name: String,
    pub source: String,
    pub dtype: DType,
    pub default_val: String,
    #[serde(default)]
    pub embed: Option<EmbedConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorDef {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub params: serde_yaml::Value,
    #[serde(default)]
    pub embed: Option<EmbedConfig>,
}

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
