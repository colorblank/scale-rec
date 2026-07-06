//! FAT (Field-Aware Transformer) — main model implementation.
//!
//! Paper: From Scaling to Structured Expressivity: Rethinking Transformers
//!        for CTR Prediction.  arXiv:2511.12081  (KDD 2026)
//!
//! Architecture:
//!   1. FeatureEmbeddings → per-field token embeddings [B, F, d]
//!   2. Input projection (if feature dims differ from model dim)
//!   3. L × FATBlock (Attention + FFN + residual)
//!   4. Sum pooling → [B, d]
//!   5. Output projection + optional deep/shared-bottom MLP
//!   6. OutputHead or MultiTaskTower → per-task logits
//!
//! Field-specific projections are pre-computed once at build time
//! from the Basis-Composed Hypernetwork (§3.4), enabling zero-overhead inference.

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::{Activation, MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::{Model, ModelExecution, ModelOutput};

use super::attention::FieldDecomposedAttention;
use super::ffn::FieldAwareFFN;
use super::hypernetwork::{BasisHypernetwork, FieldProjections};

/// One FAT block: Attention + FFN + residual.
struct FATBlock {
    attn: FieldDecomposedAttention,
    ffn: FieldAwareFFN,
}

impl FATBlock {
    fn new(
        vb: VarBuilder,
        d: usize,
        d_ff: usize,
        n_heads: usize,
        num_fields: usize,
    ) -> Result<Self> {
        let attn = FieldDecomposedAttention::new(vb.pp("attn"), d, n_heads, num_fields)?;
        let ffn = FieldAwareFFN::new(vb.pp("ffn"), d, d_ff)?;
        Ok(Self { attn, ffn })
    }

    fn forward(&self, x: &Tensor, proj: &FieldProjections) -> Result<Tensor> {
        let attn_out = self
            .attn
            .forward(x, &proj.w_q, &proj.w_k, &proj.w_v, &proj.field_pair_w)?;
        let ffn_out = self.ffn.forward(&attn_out, &proj.w_ffn1, &proj.w_ffn2)?;
        ffn_out.add(x)
    }
}

/// Field-Aware Transformer model for CTR prediction.
pub struct FATModel {
    embeddings: FeatureEmbeddings,
    input_proj: Option<Linear>,
    /// Pre-computed field-specific projections (cached from hypernetwork).
    proj: FieldProjections,
    blocks: Vec<FATBlock>,
    output_proj: Linear,
    deep: Option<Mlp>,
    shared_bottom: Option<Mlp>,
    output_head: Option<OutputHead>,
    multi_task: Option<MultiTaskTower>,
}

impl FATModel {
    /// Build with output contract (modern path).
    #[allow(clippy::too_many_arguments, non_snake_case)]
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        d: usize,
        d_ff: usize,
        num_layers: usize,
        n_heads: usize,
        m: usize,
        k: usize,
        k_top: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        contract: &OutputContract,
    ) -> Result<Self> {
        let num_fields = features.len();
        let model_dim = d;

        // Feature embeddings
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;

        // Input projection (if feature dims differ from model dim)
        let first_dim = features.first().map(|f| f.embed_dim).unwrap_or(d);
        let need_proj = features.iter().any(|f| f.embed_dim != first_dim) || first_dim != d;
        let input_proj = if need_proj {
            Some(linear(first_dim, model_dim, vb.pp("input_proj"))?)
        } else {
            None
        };

        // Hypernetwork — loads bases and routers from safetensors
        let hypernet =
            BasisHypernetwork::new(vb.pp("hypernetwork"), num_fields, d, d_ff, m, k, k_top)?;
        // Pre-compose all field-specific projections at build time
        let proj = hypernet.precompute_all()?;

        // FAT blocks — all blocks share the same field projections per the paper
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block =
                FATBlock::new(vb.pp(format!("blocks.{}", i)), d, d_ff, n_heads, num_fields)?;
            blocks.push(block);
        }

        // Output projection after sum pooling
        let output_proj = linear(model_dim, model_dim, vb.pp("output_proj"))?;

        // Optional deep MLP
        let deep = if !deep_hidden_dims.is_empty() {
            let last = *deep_hidden_dims.last().unwrap();
            Some(Mlp::new(
                vb.pp("deep"),
                model_dim,
                &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                last,
                Activation::Relu,
            )?)
        } else {
            None
        };

        let fusion_dim = deep_hidden_dims.last().copied().unwrap_or(model_dim);

        // Shared bottom MLP
        let shared_bottom = if !shared_bottom_dims.is_empty() {
            let last = *shared_bottom_dims.last().unwrap();
            Some(Mlp::new(
                vb.pp("shared_bottom"),
                fusion_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                last,
                Activation::Relu,
            )?)
        } else {
            None
        };

        let shared_dim = shared_bottom_dims.last().copied().unwrap_or(fusion_dim);

        // OutputHead from contract
        let mut representation_dims = HashMap::new();
        representation_dims.insert("shared".to_string(), shared_dim);
        let output_head = OutputHead::new(contract, &representation_dims, vb.pp("output_head"))?;

        Ok(Self {
            embeddings,
            input_proj,
            proj,
            blocks,
            output_proj,
            deep,
            shared_bottom,
            output_head: Some(output_head),
            multi_task: None,
        })
    }

    /// Build with legacy MultiTaskConfig.
    #[allow(dead_code, clippy::too_many_arguments, non_snake_case)]
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        d: usize,
        d_ff: usize,
        num_layers: usize,
        n_heads: usize,
        m: usize,
        k: usize,
        k_top: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let num_fields = features.len();
        let model_dim = d;

        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;

        let first_dim = features.first().map(|f| f.embed_dim).unwrap_or(d);
        let need_proj = features.iter().any(|f| f.embed_dim != first_dim) || first_dim != d;
        let input_proj = if need_proj {
            Some(linear(first_dim, model_dim, vb.pp("input_proj"))?)
        } else {
            None
        };

        let hypernet =
            BasisHypernetwork::new(vb.pp("hypernetwork"), num_fields, d, d_ff, m, k, k_top)?;
        let proj = hypernet.precompute_all()?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block =
                FATBlock::new(vb.pp(format!("blocks.{}", i)), d, d_ff, n_heads, num_fields)?;
            blocks.push(block);
        }

        let output_proj = linear(model_dim, model_dim, vb.pp("output_proj"))?;

        let deep = if !deep_hidden_dims.is_empty() {
            let last = *deep_hidden_dims.last().unwrap();
            Some(Mlp::new(
                vb.pp("deep"),
                model_dim,
                &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                last,
                Activation::Relu,
            )?)
        } else {
            None
        };

        let fusion_dim = deep_hidden_dims.last().copied().unwrap_or(model_dim);

        let shared_bottom = if !shared_bottom_dims.is_empty() {
            let last = *shared_bottom_dims.last().unwrap();
            Some(Mlp::new(
                vb.pp("shared_bottom"),
                fusion_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                last,
                Activation::Relu,
            )?)
        } else {
            None
        };

        let shared_dim = shared_bottom_dims.last().copied().unwrap_or(fusion_dim);

        let multi_task = MultiTaskTower::new(task_config, shared_dim, vb.pp("multi_task"))?;

        Ok(Self {
            embeddings,
            input_proj,
            proj,
            blocks,
            output_proj,
            deep,
            shared_bottom,
            output_head: None,
            multi_task: Some(multi_task),
        })
    }

    /// Compute the shared FAT backbone representation: [B, shared_dim].
    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        // [Vec of B, 1, dim_i] — per-field embeddings
        let stacked = self.embeddings.forward_stacked(x_inputs)?;
        let num_fields = stacked.len();

        // Stack along field dimension or project to uniform dim
        let x = if let Some(ref proj) = self.input_proj {
            let mut projected = Vec::with_capacity(num_fields);
            for emb in &stacked {
                let e = emb.squeeze(1)?; // [B, dim_i]
                projected.push(proj.forward(&e)?.unsqueeze(1)?); // [B, 1, d]
            }
            Tensor::cat(&projected, 1)?
        } else {
            Tensor::cat(&stacked, 1)?
        };

        // FAT blocks with pre-computed field projections
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h, &self.proj)?;
        }

        // Sum pooling over fields → [B, d]
        let pooled = h.sum(1)?;
        let pooled = self.output_proj.forward(&pooled)?;

        // Optional deep MLP
        let pooled = if let Some(ref deep) = self.deep {
            deep.forward(&pooled)?
        } else {
            pooled
        };

        // Optional shared bottom
        let pooled = if let Some(ref sb) = self.shared_bottom {
            sb.forward(&pooled)?
        } else {
            pooled
        };

        Ok(pooled)
    }
}

impl Model for FATModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return self.forward_execution(x_inputs).map(|exec| exec.outputs);
        }
        let shared = self.shared(x_inputs)?;
        if let Some(ref task) = self.multi_task {
            task.forward(&shared)
        } else {
            Err(candle_core::Error::Msg(
                "FATModel has no output configured".into(),
            ))
        }
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let shared = self.shared(x_inputs)?;
        if let Some(ref head) = self.output_head {
            let mut reps = HashMap::new();
            reps.insert("shared".to_string(), shared);
            head.forward(&reps)
        } else {
            let outputs = self.forward(x_inputs)?;
            Ok(ModelExecution::new(outputs.clone(), outputs))
        }
    }
}
