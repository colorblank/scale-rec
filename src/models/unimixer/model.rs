//! UniMixerModel：完整的 UniMixer 架构，特征分词 → Token 交互 → 多任务塔。
use super::siamese_norm::{SiameseNorm, SiameseNormOutput};
use super::tokenizer::FeatureTokenizer;
use super::unimixer_block::{BlockOutput, UniMixerBlock};
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::Model;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

pub struct UniMixerModel {
    pub embed_dim: usize,
    pub block_size: usize,
    pub use_siamese: bool,
    pub temperature: f64,
    tokenizer: FeatureTokenizer,
    blocks: Vec<UniMixerBlock>,
    task_towers: MultiTaskTower,
    final_norm: Option<SiameseNorm>,
}

impl UniMixerModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer: FeatureTokenizer,
        token_dim: usize,
        num_tokens: usize,
        num_blocks: usize,
        block_size_opt: Option<usize>,
        use_lite: bool,
        hidden_factor: f64,
        num_basis: usize,
        rank: usize,
        task_config: &MultiTaskConfig,
        use_siamese: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed_dim = num_tokens * token_dim;
        let block_size = block_size_opt.unwrap_or(token_dim);
        let mut blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let block = UniMixerBlock::new(
                embed_dim,
                block_size,
                token_dim,
                num_tokens,
                use_lite,
                hidden_factor,
                num_basis,
                rank,
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }
        let task_towers = MultiTaskTower::new(task_config, embed_dim, vb.pp("task_towers"))?;
        let final_norm = if use_siamese {
            Some(SiameseNorm::new(embed_dim, 1e-5, vb.pp("final_norm"))?)
        } else {
            None
        };
        Ok(Self {
            embed_dim,
            block_size,
            use_siamese,
            temperature: 1.0,
            tokenizer,
            blocks,
            task_towers,
            final_norm,
        })
    }

    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }

    pub fn forward_with_temperature(
        &self,
        x_inputs: &HashMap<String, Tensor>,
        temperature: f64,
    ) -> Result<HashMap<String, Tensor>> {
        let tokens = self.tokenizer.forward(x_inputs)?;
        let (batch_size, _, _) = tokens.dims3()?;
        let mut x = tokens.reshape((batch_size, self.embed_dim))?;
        let output = if self.use_siamese {
            let mut x_bar = x.clone();
            let mut y_bar = x.clone();
            for block in &self.blocks {
                let res = block.forward(&x, temperature, Some(&x_bar), Some(&y_bar), true)?;
                if let BlockOutput::Siamese(new_x_bar, new_y_bar) = res {
                    x_bar = new_x_bar;
                    y_bar = new_y_bar;
                    x = x_bar.clone();
                } else {
                    candle_core::bail!("Expected Siamese output");
                }
            }
            let final_norm = self.final_norm.as_ref().unwrap();
            let fusion_res = final_norm.forward(&x_bar, &y_bar, None)?;
            if let SiameseNormOutput::Fused(fused_out) = fusion_res {
                fused_out
            } else {
                candle_core::bail!("Expected Fused output");
            }
        } else {
            for block in &self.blocks {
                let res = block.forward(&x, temperature, None, None, false)?;
                if let BlockOutput::Standard(new_x) = res {
                    x = new_x;
                } else {
                    candle_core::bail!("Expected Standard output");
                }
            }
            x
        };
        self.task_towers.forward(&output)
    }
}

impl Model for UniMixerModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        self.forward_with_temperature(x_inputs, self.temperature)
    }
}
