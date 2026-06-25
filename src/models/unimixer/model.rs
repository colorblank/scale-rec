//! UniMixerModel：完整的 UniMixer 架构，特征分词 → Token 交互 → 多任务塔。
use super::profile;
use super::siamese_norm::{SiameseNorm, SiameseNormOutput};
use super::tokenizer::FeatureTokenizer;
use super::unimixer_block::{BlockOutput, UniMixerBlock};
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::{Model, ModelExecution, ModelOutput};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// 完整的 UniMixer 模型：特征分词 → Token 交互 → 多任务塔。
pub struct UniMixerModel {
    /// token 序列展平后的总维度。
    pub embed_dim: usize,
    /// UniMixing 分块大小。
    pub block_size: usize,
    /// 是否使用 SiameseNorm 双流结构。
    pub use_siamese: bool,
    /// Sinkhorn/归一化温度。
    pub temperature: f64,
    tokenizer: FeatureTokenizer,
    blocks: Vec<UniMixerBlock>,
    task_towers: Option<MultiTaskTower>,
    output_head: Option<OutputHead>,
    final_norm: Option<SiameseNorm>,
}

impl UniMixerModel {
    #[allow(clippy::too_many_arguments)]
    /// 构造 UniMixerModel。
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
        if use_lite && num_basis == 0 {
            candle_core::bail!("num_basis must be > 0 when use_lite=true");
        }
        if use_lite && rank == 0 {
            candle_core::bail!("rank must be > 0 when use_lite=true");
        }
        let embed_dim = num_tokens * token_dim;
        let block_size = block_size_opt.unwrap_or(token_dim);
        if block_size == 0 {
            candle_core::bail!("block_size must be > 0");
        }
        if embed_dim % block_size != 0 {
            candle_core::bail!(
                "embed_dim ({}) must be divisible by block_size ({})",
                embed_dim,
                block_size
            );
        }
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
            task_towers: Some(task_towers),
            output_head: None,
            final_norm,
        })
    }

    /// 构造使用原生输出契约的 UniMixer。
    #[allow(clippy::too_many_arguments)]
    pub fn with_output_contract(
        tokenizer: FeatureTokenizer,
        token_dim: usize,
        num_tokens: usize,
        num_blocks: usize,
        block_size_opt: Option<usize>,
        use_lite: bool,
        hidden_factor: f64,
        num_basis: usize,
        rank: usize,
        contract: &OutputContract,
        use_siamese: bool,
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
            block_size_opt,
            use_lite,
            hidden_factor,
            num_basis,
            rank,
            &empty,
            use_siamese,
            vb.clone(),
        )?;
        model.task_towers = None;
        model.output_head = Some(OutputHead::new(
            contract,
            &HashMap::from([("shared".to_string(), model.embed_dim)]),
            vb.pp("output_head"),
        )?);
        Ok(model)
    }

    /// 设置退火温度。
    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }

    /// 使用指定温度执行前向传播。
    pub fn forward_with_temperature(
        &self,
        x_inputs: &HashMap<String, Tensor>,
        temperature: f64,
    ) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self
                .forward_execution_with_temperature(x_inputs, temperature)?
                .outputs);
        }
        self.task_towers
            .as_ref()
            .unwrap()
            .forward(&self.shared_with_temperature(x_inputs, temperature)?)
    }

    fn shared_with_temperature(
        &self,
        x_inputs: &HashMap<String, Tensor>,
        temperature: f64,
    ) -> Result<Tensor> {
        if temperature <= 0.0 {
            candle_core::bail!("temperature must be > 0");
        }
        let total_timer = profile::start();
        let tokenizer_timer = profile::start();
        let tokens = self.tokenizer.forward(x_inputs)?;
        profile::log("model.tokenizer", tokenizer_timer);
        let reshape_timer = profile::start();
        let (batch_size, _, _) = tokens.dims3()?;
        let mut x = tokens.reshape((batch_size, self.embed_dim))?;
        profile::log("model.tokens_reshape", reshape_timer);
        let output = if self.use_siamese {
            let mut x_bar = x.clone();
            let mut y_bar = x.clone();
            for (idx, block) in self.blocks.iter().enumerate() {
                let block_timer = profile::start();
                let res = block.forward(&x, temperature, Some(&x_bar), Some(&y_bar), true)?;
                profile::log(&format!("model.block.{idx}.total"), block_timer);
                if let BlockOutput::Siamese(new_x_bar, new_y_bar) = res {
                    x_bar = new_x_bar;
                    y_bar = new_y_bar;
                    x = x_bar.clone();
                } else {
                    candle_core::bail!("Expected Siamese output");
                }
            }
            let final_norm_timer = profile::start();
            let final_norm = self.final_norm.as_ref().unwrap();
            let fusion_res = final_norm.forward(&x_bar, &y_bar, None)?;
            profile::log("model.final_norm", final_norm_timer);
            if let SiameseNormOutput::Fused(fused_out) = fusion_res {
                fused_out
            } else {
                candle_core::bail!("Expected Fused output");
            }
        } else {
            for (idx, block) in self.blocks.iter().enumerate() {
                let block_timer = profile::start();
                let res = block.forward(&x, temperature, None, None, false)?;
                profile::log(&format!("model.block.{idx}.total"), block_timer);
                if let BlockOutput::Standard(new_x) = res {
                    x = new_x;
                } else {
                    candle_core::bail!("Expected Standard output");
                }
            }
            x
        };
        profile::log("model.total", total_timer);
        Ok(output)
    }

    fn forward_execution_with_temperature(
        &self,
        x_inputs: &HashMap<String, Tensor>,
        temperature: f64,
    ) -> Result<ModelExecution> {
        let shared = self.shared_with_temperature(x_inputs, temperature)?;
        if let Some(head) = &self.output_head {
            return head.forward(&HashMap::from([("shared".to_string(), shared)]));
        }
        let towers_timer = profile::start();
        let outputs = self.task_towers.as_ref().unwrap().forward(&shared)?;
        profile::log("model.task_towers", towers_timer);
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}

impl Model for UniMixerModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        self.forward_with_temperature(x_inputs, self.temperature)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        self.forward_execution_with_temperature(x_inputs, self.temperature)
    }

    fn warmup(&self) -> Result<()> {
        for block in &self.blocks {
            block.warmup(self.temperature)?;
        }
        Ok(())
    }
}
