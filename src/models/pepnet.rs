//! PEPNet: Parameter and Embedding Personalized Network.
//!
//! Architecture:
//!   FeatureEmbeddings → EPNet gate → Deep MLP → shared_bottom → PPNet gate → towers
//!
//! EPNet computes an element-wise gate on the concatenated embedding vector,
//! conditioned on a learned prior representation (aggregated from all feature embeddings).
//! PPNet applies a similar gate to the shared representation before task towers.

use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::{apply_relation, Activation, MultiTaskConfig, TaskRelation, TaskTower};
use candle_core::{Result, Tensor};
use candle_nn::{linear, linear_no_bias, Module, VarBuilder};
use std::collections::HashMap;

pub struct PEPNet {
    embeddings: FeatureEmbeddings,
    /// Projects feature-wise pooled prior → fixed-dim prior representation.
    prior_proj: candle_nn::Linear,
    /// EPNet gate: prior_rep → [total_dim] sigmoid gate applied to embeddings.
    epnet_gate: candle_nn::Linear,
    /// PPNet gate: prior_rep → [shared_dim] sigmoid gate applied to shared bottom.
    ppnet_gate: candle_nn::Linear,
    deep: Option<Mlp>,
    shared_bottom: Option<Mlp>,
    towers: Vec<(String, TaskTower)>,
    relations: Vec<TaskRelation>,
    output_head: Option<OutputHead>,
}

impl PEPNet {
    fn build_gate(vb: VarBuilder, prior_dim: usize, gate_dim: usize) -> Result<candle_nn::Linear> {
        linear_no_bias(prior_dim, gate_dim, vb)
    }

    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        prior_dim: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;

        let prior_proj = linear(embeddings.num_features, prior_dim, vb.pp("prior_proj"))?;
        let epnet_gate = Self::build_gate(vb.pp("epnet_gate"), prior_dim, total_dim)?;
        let ppnet_gate = Self::build_gate(
            vb.pp("ppnet_gate"),
            prior_dim,
            shared_bottom_dims.last().copied().unwrap_or(total_dim),
        )?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, total_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("deep"),
                total_dim,
                &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), output_dim)
        };

        let (shared_bottom, shared_dim) = if shared_bottom_dims.is_empty() {
            (None, fusion_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("shared_bottom"),
                fusion_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), output_dim)
        };

        let mut towers = Vec::with_capacity(task_config.towers.len());
        for tower_config in &task_config.towers {
            towers.push((
                tower_config.name.clone(),
                TaskTower::new(
                    tower_config,
                    shared_dim,
                    vb.pp(format!("{}_tower", tower_config.name)),
                )?,
            ));
        }

        Ok(Self {
            embeddings,
            prior_proj,
            epnet_gate,
            ppnet_gate,
            deep,
            shared_bottom,
            towers,
            relations: task_config.relations.clone(),
            output_head: None,
        })
    }

    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        prior_dim: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        contract: &OutputContract,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;

        let prior_proj = linear(embeddings.num_features, prior_dim, vb.pp("prior_proj"))?;
        let epnet_gate = Self::build_gate(vb.pp("epnet_gate"), prior_dim, total_dim)?;
        let ppnet_gate = Self::build_gate(
            vb.pp("ppnet_gate"),
            prior_dim,
            shared_bottom_dims.last().copied().unwrap_or(total_dim),
        )?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, total_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("deep"),
                    total_dim,
                    &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                output_dim,
            )
        };

        let (shared_bottom, shared_dim) = if shared_bottom_dims.is_empty() {
            (None, fusion_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("shared_bottom"),
                    fusion_dim,
                    &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                output_dim,
            )
        };

        let output_head = OutputHead::new(
            contract,
            &HashMap::from([("shared".to_string(), shared_dim)]),
            vb.pp("output_head"),
        )?;

        Ok(Self {
            embeddings,
            prior_proj,
            epnet_gate,
            ppnet_gate,
            deep,
            shared_bottom,
            towers: vec![],
            relations: vec![],
            output_head: Some(output_head),
        })
    }

    /// Build the shared representation: embeddings → EPNet gate → deep → shared_bottom → PPNet gate.
    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let stacked = self.embeddings.forward_stacked(x_inputs)?;
        // stacked: Vec<[batch, 1, dim_i]>

        // Compute prior: mean-pool each per-feature embedding → [batch, num_features].
        let mut prior_parts = Vec::with_capacity(stacked.len());
        for emb in &stacked {
            // emb: [batch, 1, dim_i]
            let pooled = emb.mean_keepdim(2)?; // [batch, 1, 1]
            prior_parts.push(pooled.squeeze(2)?); // [batch, 1]
        }
        let prior_raw = Tensor::cat(&prior_parts, 1)?; // [batch, num_features]
        let prior = self.prior_proj.forward(&prior_raw)?; // [batch, prior_dim]

        // EPNet gate on embeddings
        let epnet_scale = self.epnet_gate.forward(&prior)?; // [batch, total_dim]
        let epnet_scale = candle_nn::ops::sigmoid(&epnet_scale)?;
        let dense_concat = Tensor::cat(
            &stacked
                .iter()
                .map(|e| e.squeeze(1))
                .collect::<Result<Vec<_>>>()?,
            1,
        )?; // [batch, total_dim]
        let gated = dense_concat.broadcast_mul(&epnet_scale)?;

        let mut shared = match &self.deep {
            Some(deep) => deep.forward(&gated)?,
            None => gated,
        };
        if let Some(shared_bottom) = &self.shared_bottom {
            shared = shared_bottom.forward(&shared)?;
        }

        // PPNet gate on shared representation
        let ppnet_scale = self.ppnet_gate.forward(&prior)?; // [batch, shared_dim]
        let ppnet_scale = candle_nn::ops::sigmoid(&ppnet_scale)?;
        let gated_shared = shared.broadcast_mul(&ppnet_scale)?;

        Ok(gated_shared)
    }
}

impl Model for PEPNet {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let shared = self.shared(x_inputs)?;
        let mut outputs = ModelOutput::new();
        for (name, tower) in &self.towers {
            outputs.insert_binary_logit(name.clone(), tower.forward(&shared)?);
        }
        for relation in &self.relations {
            outputs
                .insert_probability(relation.target.clone(), apply_relation(relation, &outputs)?);
        }
        Ok(outputs)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        if let Some(head) = &self.output_head {
            return head.forward(&HashMap::from([(
                "shared".to_string(),
                self.shared(x_inputs)?,
            )]));
        }
        let outputs = self.forward(x_inputs)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}
