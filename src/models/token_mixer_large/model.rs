//! TokenMixerLargeModel: Tokenizer + M TokenMixerLargeBlocks + MultiTaskTower.
use super::block::TokenMixerLargeBlock;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use crate::models::{Model, ModelOutput};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// TokenMixer-Large model: FeatureTokenizer + M blocks + task towers.
pub struct TokenMixerLargeModel {
    /// token 序列展平后的总维度。
    pub embed_dim: usize,
    tokenizer: FeatureTokenizer,
    blocks: Vec<TokenMixerLargeBlock>,
    task_towers: MultiTaskTower,
}

impl TokenMixerLargeModel {
    /// Construct a TokenMixer-Large model.
    pub fn new(
        tokenizer: FeatureTokenizer,
        token_dim: usize,
        num_tokens: usize,
        num_blocks: usize,
        num_heads: usize,
        hidden_factor: f64,
        task_config: &MultiTaskConfig,
        vb: VarBuilder,
        down_init_scale: f64,
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
        if hidden_factor <= 0.0 {
            candle_core::bail!("hidden_factor must be > 0");
        }
        if num_heads == 0 {
            candle_core::bail!("num_heads must be > 0");
        }
        let embed_dim = num_tokens * token_dim;
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let block = TokenMixerLargeBlock::new(
                embed_dim,
                token_dim,
                num_tokens,
                num_heads,
                hidden_factor,
                vb.pp(format!("blocks.{}", i)),
                down_init_scale,
            )?;
            blocks.push(block);
        }
        let task_towers = MultiTaskTower::new(task_config, embed_dim, vb.pp("task_towers"))?;
        Ok(Self {
            embed_dim,
            tokenizer,
            blocks,
            task_towers,
        })
    }
}

impl Model for TokenMixerLargeModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        let tokens = self.tokenizer.forward(x_inputs)?;
        let (batch_size, _, _) = tokens.dims3()?;
        let mut x = tokens.reshape((batch_size, self.embed_dim))?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.task_towers.forward(&x)
    }
}
