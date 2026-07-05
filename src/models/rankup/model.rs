//! RankUp model: RankMixer backbone over RankUp tokens plus task-token-aware heads.

use super::tokenizer::{CrossTokenConfig, RankUpTokenizer};
use crate::layers::embedding::FeatureSpec;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::rankmixer::block::RankMixerBlock;
use crate::models::{Model, ModelExecution, ModelOutput};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// Config for RankUp construction.
#[derive(Debug, Clone)]
pub struct RankUpConfig {
    /// Per-token latent dimension.
    pub token_dim: usize,
    /// Number of shuffled sparse feature groups.
    pub num_sparse_tokens: usize,
    /// Number of backbone blocks.
    pub num_blocks: usize,
    /// Token-mixing head count. For this RankMixer-style block it must equal total tokens.
    pub num_heads: Option<usize>,
    /// Per-token FFN hidden multiplier.
    pub hidden_factor: f64,
    /// Deterministic feature permutation seed.
    pub permutation_seed: u64,
    /// Number of independent embedding tables per feature.
    pub multi_embedding_tables: usize,
    /// Append a global token over all grouped embeddings.
    pub use_global_token: bool,
    /// Optional learned feature cross token.
    pub cross_token: Option<CrossTokenConfig>,
    /// Number of task-specific tokens.
    pub num_task_tokens: usize,
}

/// RankUp model.
pub struct RankUpModel {
    tokenizer: RankUpTokenizer,
    blocks: Vec<RankMixerBlock>,
    task_towers: Option<MultiTaskTower>,
    output_head: Option<OutputHead>,
}

impl RankUpModel {
    /// Construct with legacy task towers.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: RankUpConfig,
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let tokenizer = RankUpTokenizer::new(
            vb.pp("tokenizer"),
            features,
            config.token_dim,
            config.num_sparse_tokens,
            config.permutation_seed,
            config.multi_embedding_tables,
            config.use_global_token,
            config.cross_token.clone(),
            config.num_task_tokens,
        )?;
        let blocks = build_blocks(&tokenizer, &config, vb.pp("blocks"))?;
        let task_towers = MultiTaskTower::new(task_config, config.token_dim, vb.pp("task_towers"))?;
        Ok(Self {
            tokenizer,
            blocks,
            task_towers: Some(task_towers),
            output_head: None,
        })
    }

    /// Construct with a native output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        mut config: RankUpConfig,
        contract: &OutputContract,
    ) -> Result<Self> {
        if config.num_task_tokens == 0 {
            config.num_task_tokens = contract.graph.towers.len();
        }
        let tokenizer = RankUpTokenizer::new(
            vb.pp("tokenizer"),
            features,
            config.token_dim,
            config.num_sparse_tokens,
            config.permutation_seed,
            config.multi_embedding_tables,
            config.use_global_token,
            config.cross_token.clone(),
            config.num_task_tokens,
        )?;
        let blocks = build_blocks(&tokenizer, &config, vb.pp("blocks"))?;
        let mut representation_dims = HashMap::from([("shared".to_string(), config.token_dim)]);
        for idx in 0..tokenizer.num_task_tokens() {
            representation_dims.insert(format!("task_{idx}"), config.token_dim * 2);
        }
        let output_head = OutputHead::new(contract, &representation_dims, vb.pp("output_head"))?;
        Ok(Self {
            tokenizer,
            blocks,
            task_towers: None,
            output_head: Some(output_head),
        })
    }

    fn representations(
        &self,
        x_inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut x = self.tokenizer.forward(x_inputs)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let total_tokens = x.dim(1)?;
        let task_tokens = self.tokenizer.num_task_tokens();
        let shared_tokens = total_tokens - task_tokens;
        let shared = x.narrow(1, 0, shared_tokens)?.mean(1)?;
        let mut representations = HashMap::from([("shared".to_string(), shared.clone())]);
        if task_tokens > 0 {
            let task_slice = x.narrow(1, shared_tokens, task_tokens)?;
            for idx in 0..task_tokens {
                let task = task_slice.narrow(1, idx, 1)?.squeeze(1)?;
                representations.insert(format!("task_{idx}"), Tensor::cat(&[&task, &shared], 1)?);
            }
        }
        Ok(representations)
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut representations = self.representations(x_inputs)?;
        representations.remove("shared").ok_or_else(|| {
            candle_core::Error::Msg("RankUp shared representation is missing".into())
        })
    }
}

impl Model for RankUpModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        self.task_towers
            .as_ref()
            .unwrap()
            .forward(&self.shared(x_inputs)?)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        if let Some(head) = &self.output_head {
            return head.forward(&self.representations(x_inputs)?);
        }
        let outputs = self
            .task_towers
            .as_ref()
            .unwrap()
            .forward(&self.shared(x_inputs)?)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}

fn build_blocks(
    tokenizer: &RankUpTokenizer,
    config: &RankUpConfig,
    vb: VarBuilder,
) -> Result<Vec<RankMixerBlock>> {
    if config.num_blocks == 0 {
        candle_core::bail!("num_blocks must be > 0");
    }
    if config.hidden_factor <= 0.0 {
        candle_core::bail!("hidden_factor must be > 0");
    }
    let num_heads = config.num_heads.unwrap_or(tokenizer.num_tokens);
    let mut blocks = Vec::with_capacity(config.num_blocks);
    for i in 0..config.num_blocks {
        blocks.push(RankMixerBlock::new(
            config.token_dim,
            tokenizer.num_tokens,
            num_heads,
            config.hidden_factor,
            vb.pp(i.to_string()),
        )?);
    }
    Ok(blocks)
}
