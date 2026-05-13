use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelConfig {
    LR,
    DeepFM {
        fm_k: usize,
        #[serde(default)]
        deep_hidden_dims: Vec<usize>,
    },
    MMoE {
        #[serde(default)]
        shared_bottom_dims: Vec<usize>,
        num_experts: usize,
        #[serde(default)]
        expert_hidden_dims: Vec<usize>,
        expert_output_dim: usize,
        task_configs: Vec<TaskConfigEntry>,
    },
    ESMM {
        #[serde(default)]
        shared_bottom_dims: Vec<usize>,
        #[serde(default)]
        ctr_hidden_dims: Vec<usize>,
        #[serde(default)]
        cvr_hidden_dims: Vec<usize>,
    },
    UniMixer {
        token_dim: usize,
        num_tokens: usize,
        num_blocks: usize,
        #[serde(default)]
        block_size: Option<usize>,
        #[serde(default)]
        use_lite: bool,
        #[serde(default = "default_hidden_factor")]
        hidden_factor: f64,
        #[serde(default = "default_num_basis")]
        num_basis: usize,
        #[serde(default = "default_rank")]
        rank: usize,
        #[serde(default)]
        use_siamese: bool,
        task_config: MultiTaskConfig,
    },
}

fn default_hidden_factor() -> f64 { 1.0 }
fn default_num_basis() -> usize { 4 }
fn default_rank() -> usize { 16 }

impl ModelConfig {
    /// Build model from config. Feature specs come from `FeatureDag::embeddable_features()`.
    pub fn build(
        &self,
        vb: VarBuilder,
        features: &[(String, usize, usize)],
        tokenizer: Option<FeatureTokenizer>,
    ) -> Result<Box<dyn Model>> {
        match self {
            ModelConfig::LR => {
                let model = lr::LogisticRegression::new(vb, features)?;
                Ok(Box::new(model))
            }
            ModelConfig::DeepFM { fm_k, deep_hidden_dims } => {
                let model = deepfm::DeepFM::new(vb, features, *fm_k, deep_hidden_dims)?;
                Ok(Box::new(model))
            }
            ModelConfig::MMoE { shared_bottom_dims, num_experts, expert_hidden_dims, expert_output_dim, task_configs } => {
                let task_cfgs: Vec<(String, Vec<usize>)> = task_configs.iter().map(|t| (t.name.clone(), t.tower_dims.clone())).collect();
                let model = mmoe::MMoE::new(vb, features, shared_bottom_dims, *num_experts, expert_hidden_dims, *expert_output_dim, &task_cfgs)?;
                Ok(Box::new(model))
            }
            ModelConfig::ESMM { shared_bottom_dims, ctr_hidden_dims, cvr_hidden_dims } => {
                let model = esmm::ESMM::new(vb, features, shared_bottom_dims, ctr_hidden_dims, cvr_hidden_dims)?;
                Ok(Box::new(model))
            }
            ModelConfig::UniMixer { token_dim, num_tokens, num_blocks, block_size, use_lite, hidden_factor, num_basis, rank, use_siamese, task_config } => {
                let tokenizer = tokenizer.ok_or_else(|| {
                    candle_core::Error::Msg("UniMixer requires external FeatureTokenizer from FeatureDag".into())
                })?;
                let model = unimixer::model::UniMixerModel::new(
                    tokenizer, *token_dim, *num_tokens, *num_blocks, *block_size,
                    *use_lite, *hidden_factor, *num_basis, *rank, task_config, *use_siamese, vb.pp("unimixer"),
                )?;
                Ok(Box::new(model))
            }
        }
    }

    pub fn model_type(&self) -> &str {
        match self {
            ModelConfig::LR => "lr",
            ModelConfig::DeepFM { .. } => "deepfm",
            ModelConfig::MMoE { .. } => "mmoe",
            ModelConfig::ESMM { .. } => "esmm",
            ModelConfig::UniMixer { .. } => "unimixer",
        }
    }
}
