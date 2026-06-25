//! RankMixer model: tokenizer + blocks + mean pooling + task towers.

use super::block::RankMixerBlock;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use crate::models::{Model, ModelExecution, ModelOutput};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// Dense RankMixer model with mean-pooled token output.
pub struct RankMixerModel {
    tokenizer: FeatureTokenizer,
    blocks: Vec<RankMixerBlock>,
    task_towers: Option<MultiTaskTower>,
    output_head: Option<OutputHead>,
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
            task_towers: Some(task_towers),
            output_head: None,
        })
    }

    /// Construct a RankMixer model with a native output contract.
    pub fn with_output_contract(
        tokenizer: FeatureTokenizer,
        token_dim: usize,
        num_tokens: usize,
        num_blocks: usize,
        num_heads: usize,
        hidden_factor: f64,
        contract: &OutputContract,
        vb: VarBuilder,
    ) -> Result<Self> {
        let empty = MultiTaskConfig {
            towers: vec![],
            relations: vec![],
        };
        let mut model = Self::new(
            tokenizer,
            token_dim,
            num_tokens,
            num_blocks,
            num_heads,
            hidden_factor,
            &empty,
            vb.clone(),
        )?;
        model.task_towers = None;
        model.output_head = Some(OutputHead::new(
            contract,
            &HashMap::from([("shared".to_string(), token_dim)]),
            vb.pp("output_head"),
        )?);
        Ok(model)
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut x = self.tokenizer.forward(x_inputs)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        x.mean(1)
    }
}

impl Model for RankMixerModel {
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
        let shared = self.shared(x_inputs)?;
        if let Some(head) = &self.output_head {
            return head.forward(&HashMap::from([("shared".to_string(), shared)]));
        }
        let outputs = self.task_towers.as_ref().unwrap().forward(&shared)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}
