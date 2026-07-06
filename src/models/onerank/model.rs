//! OneRank: Unified Transformer-Native Ranking Architecture — main model.
//!
//! Paper: arXiv:2606.16838  (KDD 2026)
//!
//! Architecture:
//!   1. FeatureEmbeddings → per-field tokens  [B, F, d]
//!   2. Task token injection → [B, F+K, d] with structured mask
//!   3. L × OneRankBlock (Pre-LN MHSA + Pre-LN FFN)
//!   4. Task extraction + feature pooling → SD projection
//!   5. Cross-task attention (configurable mask)
//!   6. Dynamic matching scoring: s_k = z_k^T · r_k

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::models::output_contract::OutputContract;
use crate::models::{Model, ModelExecution, ModelOutput, OutputKind};

use super::encoding::{build_attention_mask, OneRankBlock};
use super::prediction::{matching_score, CrossTaskAttention};

/// OneRank model for multi-task ranking.
pub struct OneRankModel {
    embeddings: FeatureEmbeddings,
    input_proj: Option<Linear>,
    /// [K, d] learnable task token parameters.
    task_tokens: Tensor,
    /// [1, N_max, d] learned positional encoding.
    pos_encoding: Tensor,
    blocks: Vec<OneRankBlock>,
    sd_proj: Linear,
    cross_task: CrossTaskAttention,
    #[allow(dead_code)]
    num_fields: usize,
    #[allow(dead_code)]
    num_tasks: usize,
    #[allow(dead_code)]
    d: usize,
    task_names: Vec<String>,
}

impl OneRankModel {
    /// Build OneRank with output contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        d: usize,
        d_ff: usize,
        num_layers: usize,
        n_heads: usize,
        num_tasks: usize,
        cross_task_mask: &str,
        contract: &OutputContract,
    ) -> Result<Self> {
        let num_fields = features.len();

        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;

        let first_dim = features.first().map(|f| f.embed_dim).unwrap_or(d);
        let need_proj = features.iter().any(|f| f.embed_dim != first_dim) || first_dim != d;
        let input_proj = if need_proj {
            Some(linear(first_dim, d, vb.pp("input_proj"))?)
        } else {
            None
        };

        let task_tokens = vb.get_with_hints(
            (num_tasks, d),
            "task_tokens",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        let pos_encoding = vb.get_with_hints(
            (1, num_fields + num_tasks, d),
            "pos_encoding",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = OneRankBlock::new(vb.pp(format!("blocks.{}", i)), d, d_ff, n_heads)?;
            blocks.push(block);
        }

        let sd_proj = linear(d, d, vb.pp("sd_proj.1"))?;

        let cross_task =
            CrossTaskAttention::new(vb.pp("cross_task"), d, num_tasks, cross_task_mask)?;

        let task_names: Vec<String> = contract
            .graph
            .towers
            .iter()
            .map(|t| t.name.clone())
            .collect();

        Ok(Self {
            embeddings,
            input_proj,
            task_tokens,
            pos_encoding,
            blocks,
            sd_proj,
            cross_task,
            num_fields,
            num_tasks,
            d,
            task_names,
        })
    }

    /// Forward: compute matching scores and produce ModelOutput.
    pub fn forward_impl(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        let scores = self.compute_scores(x_inputs)?;
        let mut output = ModelOutput::new();
        for (k, name) in self.task_names.iter().enumerate() {
            let s = scores.narrow(1, k, 1)?.squeeze(1)?;
            output.insert(name.clone(), s, OutputKind::BinaryLogit);
        }
        Ok(output)
    }

    /// Compute per-task matching scores: [B, K]
    fn compute_scores(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let stacked = self.embeddings.forward_stacked(x_inputs)?;
        let num_fields = stacked.len();

        let x = if let Some(ref proj) = self.input_proj {
            let mut projected = Vec::with_capacity(num_fields);
            for emb in &stacked {
                let e = emb.squeeze(1)?;
                projected.push(proj.forward(&e)?.unsqueeze(1)?);
            }
            Tensor::cat(&projected, 1)?
        } else {
            Tensor::cat(&stacked, 1)?
        };

        let (b, f, d) = x.dims3()?;
        let k = self.num_tasks;

        // Inject task tokens: [B, K, d]
        let task_tokens = self.task_tokens.unsqueeze(0)?.expand((b, k, d))?;
        let h = Tensor::cat(&[&x, &task_tokens], 1)?;

        // Add positional encoding
        let h = h.broadcast_add(&self.pos_encoding)?;

        // Build structured attention mask
        let mask = build_attention_mask(f, k, x.device())?;

        // Transformer blocks
        let mut h = h;
        for block in &self.blocks {
            h = block.forward(&h, &mask)?;
        }

        // Extract task representations: [B, K, d]
        let task_repr = h.narrow(1, f, k)?;

        // Feature pool → SD: [B, d]
        let feat_pool = h.narrow(1, 0, f)?.mean(1)?;
        let sd = self.sd_proj.forward(&feat_pool)?;
        let sd = sd.unsqueeze(1)?.expand((b, k, d))?;

        // Cross-task attention
        let cross_input = (task_repr.clone() + sd)?;
        let global_repr = self.cross_task.forward(&cross_input)?;

        // Matching scoring: [B, K]
        matching_score(&global_repr, &task_repr)
    }
}

impl Model for OneRankModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        self.forward_impl(x_inputs)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let outputs = self.forward_impl(x_inputs)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}
