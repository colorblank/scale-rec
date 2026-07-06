//! MixFormer: Co-Scaling Up Dense and Sequence — main model.
//!
//! Paper: arXiv:2602.14110  (KDD 2026)
//!
//! Architecture:
//!   1. FeatureEmbeddings → concat → project → [B, N, D] (N query heads)
//!   2. L × MixFormerBlock (QueryMixer → OutputFusion)
//!   3. Mean pool heads → [B, D] → output projection → OutputHead

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Linear, VarBuilder};

use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::{Model, ModelExecution, ModelOutput};

use super::encoding::MixFormerBlock;

/// MixFormer model for multi-task ranking.
pub struct MixFormerModel {
    embeddings: FeatureEmbeddings,
    /// Projects concatenated features to [B, N*D].
    input_proj: Linear,
    blocks: Vec<MixFormerBlock>,
    output_norm: candle_nn::LayerNorm,
    output_linear: Linear,
    output_head: Option<OutputHead>,
    num_heads: usize,
    d: usize,
}

impl MixFormerModel {
    /// Build MixFormer with output contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        d: usize,
        d_ff: usize,
        num_heads: usize,
        num_layers: usize,
        contract: &OutputContract,
    ) -> Result<Self> {
        if d % num_heads != 0 {
            candle_core::bail!(
                "MixFormer: d ({}) must be divisible by num_heads ({}) for head_mixing",
                d,
                num_heads
            );
        }
        let total_dim: usize = features.iter().map(|f| f.embed_dim).sum();

        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let input_proj = linear(total_dim, num_heads * d, vb.pp("input_proj"))?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = MixFormerBlock::new(vb.pp(format!("blocks.{}", i)), d, d_ff, num_heads)?;
            blocks.push(block);
        }

        let output_norm = layer_norm(d, 1e-5, vb.pp("output_proj.0"))?;
        let output_linear = linear(d, d, vb.pp("output_proj.1"))?;

        let mut representation_dims = HashMap::new();
        representation_dims.insert("shared".to_string(), d);
        let output_head = OutputHead::new(contract, &representation_dims, vb.pp("output_head"))?;

        Ok(Self {
            embeddings,
            input_proj,
            blocks,
            output_norm,
            output_linear,
            output_head: Some(output_head),
            num_heads,
            d,
        })
    }

    /// Compute shared representation: [B, D]
    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let stacked = self.embeddings.forward_stacked(x_inputs)?;

        // Flatten per-field embeddings: [B, 1, dim_i] → [B, dim_i] → cat → [B, total_dim]
        let mut flat = Vec::with_capacity(stacked.len());
        for emb in &stacked {
            flat.push(emb.squeeze(1)?);
        }
        let x = Tensor::cat(&flat, 1)?;

        // Project to multi-head query space
        let x = self.input_proj.forward(&x)?; // [B, N*D]
        let (b, _) = x.dims2()?;
        let x = x.reshape((b, self.num_heads, self.d))?; // [B, N, D]

        // MixFormer blocks
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }

        // Pool heads → [B, D]
        let h = h.mean(1)?;
        let h = self.output_norm.forward(&h)?;
        self.output_linear.forward(&h)
    }
}

impl Model for MixFormerModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if let Some(ref head) = self.output_head {
            let shared = self.shared(x_inputs)?;
            let mut reps = HashMap::new();
            reps.insert("shared".to_string(), shared);
            let exec = head.forward(&reps)?;
            Ok(exec.outputs)
        } else {
            Err(candle_core::Error::Msg(
                "MixFormer has no output configured".into(),
            ))
        }
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let shared = self.shared(x_inputs)?;
        let mut reps = HashMap::new();
        reps.insert("shared".to_string(), shared);
        if let Some(ref head) = self.output_head {
            head.forward(&reps)
        } else {
            Err(candle_core::Error::Msg(
                "MixFormer has no output configured".into(),
            ))
        }
    }
}
