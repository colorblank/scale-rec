//! RankMixer model: tokenizer + blocks + mean pooling + task towers.

use super::block::RankMixerBlock;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use crate::models::Model;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// Dense RankMixer model with mean-pooled token output.
pub struct RankMixerModel {
    tokenizer: FeatureTokenizer,
    blocks: Vec<RankMixerBlock>,
    task_towers: MultiTaskTower,
}

impl RankMixerModel {
    /// Construct a RankMixer model.
    pub fn new(
        tokenizer: FeatureTokenizer,
        token_dim: usize,
        num_tokens: usize,
        num_blocks: usize,
        num_heads: usize,
        hidden_factor: f64,
        task_config: &MultiTaskConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        if token_dim == 0 {
            candle_core::bail!("token_dim must be > 0");
        }
        if num_tokens == 0 {
            candle_core::bail!("num_tokens must be > 0");
        }
        if num_blocks == 0 {
            candle_core::bail!("num_blocks must be > 0");
        }
        if num_heads == 0 {
            candle_core::bail!("num_heads must be > 0");
        }
        if hidden_factor <= 0.0 {
            candle_core::bail!("hidden_factor must be > 0");
        }
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            blocks.push(RankMixerBlock::new(
                token_dim,
                num_tokens,
                num_heads,
                hidden_factor,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }
        let task_towers = MultiTaskTower::new(task_config, token_dim, vb.pp("task_towers"))?;
        Ok(Self {
            tokenizer,
            blocks,
            task_towers,
        })
    }
}

impl Model for RankMixerModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut x = self.tokenizer.forward(x_inputs)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let pooled = x.mean(1)?;
        self.task_towers.forward(&pooled)
    }
}
