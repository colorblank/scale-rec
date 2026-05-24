//! 模型注册中心：Model trait、ModelConfig、工厂函数注册表。
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::layers::embedding::FeatureSpec;
use crate::layers::towers::MultiTaskConfig;
use crate::models::unimixer::tokenizer::FeatureTokenizer;

pub mod deepfm;
pub mod esmm;
pub mod lr;
pub mod mmoe;
pub mod unimixer;

pub trait Model: Send + Sync {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfigEntry {
    pub name: String,
    pub tower_dims: Vec<usize>,
}

/// 模型配置：类型标签 + 透传的 YAML params。
/// 每个模型自行解析 params，不再集中在中央枚举。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(rename = "type")]
    pub model_type: String,
    #[serde(default, flatten)]
    pub params: serde_yaml::Value,
}

/// Build function signature: (vb, features, tokenizer, params) -> Box<dyn Model>
type BuildFn = fn(
    VarBuilder,
    &[FeatureSpec],
    Option<FeatureTokenizer>,
    &serde_yaml::Value,
) -> Result<Box<dyn Model>>;

static REGISTRY: LazyLock<HashMap<&'static str, BuildFn>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, BuildFn> = HashMap::new();
    m.insert("lr", build_lr);
    m.insert("deepfm", build_deepfm);
    m.insert("mmoe", build_mmoe);
    m.insert("esmm", build_esmm);
    m.insert("unimixer", build_unimixer);
    m
});

impl ModelConfig {
    pub fn build(
        &self,
        vb: VarBuilder,
        features: &[FeatureSpec],
        tokenizer: Option<FeatureTokenizer>,
    ) -> Result<Box<dyn Model>> {
        match REGISTRY.get(self.model_type.as_str()) {
            Some(build_fn) => build_fn(vb, features, tokenizer, &self.params),
            None => candle_core::bail!(
                "Unknown model type: {}. Registered: {:?}",
                self.model_type,
                REGISTRY.keys().collect::<Vec<_>>()
            ),
        }
    }

    pub fn model_type(&self) -> &str {
        &self.model_type
    }
}

// ── per-model build functions ──

fn build_lr(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    _params: &serde_yaml::Value,
) -> Result<Box<dyn Model>> {
    Ok(Box::new(lr::LogisticRegression::new(vb, features)?))
}

fn build_deepfm(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
) -> Result<Box<dyn Model>> {
    let fm_k = yaml_usize(params, "fm_k", 16);
    let deep_hidden_dims: Vec<usize> = yaml_usize_seq(params, "deep_hidden_dims");
    Ok(Box::new(deepfm::DeepFM::new(
        vb,
        features,
        fm_k,
        &deep_hidden_dims,
    )?))
}

fn build_mmoe(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
) -> Result<Box<dyn Model>> {
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    let num_experts = yaml_usize(params, "num_experts", 4);
    let expert_hidden_dims: Vec<usize> = yaml_usize_seq(params, "expert_hidden_dims");
    let expert_output_dim = yaml_usize(params, "expert_output_dim", 32);
    let task_configs = parse_task_config_entries(params);
    let task_cfgs: Vec<(String, Vec<usize>)> = task_configs
        .iter()
        .map(|t| (t.name.clone(), t.tower_dims.clone()))
        .collect();
    Ok(Box::new(mmoe::MMoE::new(
        vb,
        features,
        &shared_bottom_dims,
        num_experts,
        &expert_hidden_dims,
        expert_output_dim,
        &task_cfgs,
    )?))
}

fn build_esmm(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
) -> Result<Box<dyn Model>> {
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    let click_hidden_dims: Vec<usize> = yaml_usize_seq(params, "click_hidden_dims");
    let cvr_hidden_dims: Vec<usize> = yaml_usize_seq(params, "cvr_hidden_dims");
    let detail_hidden_dims: Vec<usize> = yaml_usize_seq(params, "detail_hidden_dims");
    let stock_hidden_dims: Vec<usize> = yaml_usize_seq(params, "stock_hidden_dims");
    let stay_hidden_dims: Vec<usize> = yaml_usize_seq(params, "stay_hidden_dims");
    Ok(Box::new(esmm::ESMM::new(
        vb,
        features,
        &shared_bottom_dims,
        &click_hidden_dims,
        &cvr_hidden_dims,
        &detail_hidden_dims,
        &stock_hidden_dims,
        &stay_hidden_dims,
    )?))
}

fn build_unimixer(
    vb: VarBuilder,
    _features: &[FeatureSpec],
    tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
) -> Result<Box<dyn Model>> {
    let tokenizer = tokenizer.ok_or_else(|| {
        candle_core::Error::Msg("UniMixer requires external FeatureTokenizer".into())
    })?;
    let token_dim = yaml_usize(params, "token_dim", 64);
    let num_tokens = yaml_usize(params, "num_tokens", 8);
    let num_blocks = yaml_usize(params, "num_blocks", 2);
    let block_size = yaml_usize_opt(params, "block_size");
    let use_lite = yaml_bool(params, "use_lite");
    let hidden_factor = yaml_f64(params, "hidden_factor", 1.0);
    let num_basis = yaml_usize(params, "num_basis", 4);
    let rank = yaml_usize(params, "rank", 16);
    let use_siamese = yaml_bool(params, "use_siamese");
    let task_config = parse_multi_task_config(params);
    Ok(Box::new(unimixer::model::UniMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        num_blocks,
        block_size,
        use_lite,
        hidden_factor,
        num_basis,
        rank,
        &task_config,
        use_siamese,
        vb.pp("unimixer"),
    )?))
}

// ── YAML param helpers ──

fn yaml_usize(v: &serde_yaml::Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(default)
}

fn yaml_usize_opt(v: &serde_yaml::Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|v| v.as_u64()).map(|n| n as usize)
}

fn yaml_usize_seq(v: &serde_yaml::Value, key: &str) -> Vec<usize> {
    v.get(key)
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default()
}

fn yaml_f64(v: &serde_yaml::Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

fn yaml_bool(v: &serde_yaml::Value, key: &str) -> bool {
    v.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn parse_task_config_entries(params: &serde_yaml::Value) -> Vec<TaskConfigEntry> {
    params
        .get("task_configs")
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|entry| {
                    let name = entry.get("name")?.as_str()?.to_string();
                    let tower_dims: Vec<usize> = yaml_usize_seq(entry, "tower_dims");
                    Some(TaskConfigEntry { name, tower_dims })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_multi_task_config(params: &serde_yaml::Value) -> MultiTaskConfig {
    let task_config = params.get("task_config");
    serde_yaml::from_value::<MultiTaskConfig>(task_config.cloned().unwrap_or_default())
        .unwrap_or_else(|_| MultiTaskConfig {
            towers: vec![],
            relations: vec![],
        })
}
