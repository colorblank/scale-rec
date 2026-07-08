//! Deep Interest Network (DIN, arXiv:1706.06978).
//!
//! DIN uses a local activation unit to adaptively learn user interest
//! representation from behavior sequences with respect to a candidate ad.
//! Shared item embedding table for both behavior items and candidate ad.

use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::{embedding, Module, VarBuilder};
use std::collections::HashMap;

/// Deep Interest Network model.
pub struct DIN {
    item_embedding: candle_nn::Embedding,
    embeddings: Option<FeatureEmbeddings>,
    activation_unit: Mlp,
    mlp: Mlp,
    output_head: Option<OutputHead>,
    embed_dim: usize,
    behavior_feature: String,
    candidate_feature: String,
}

impl DIN {
    /// Construct DIN with shared item embedding + activation unit + MLP.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        item_vocab_size: usize,
        embed_dim: usize,
        activation_hidden_dims: &[usize],
        mlp_hidden_dims: &[usize],
        behavior_feature: &str,
        candidate_feature: &str,
    ) -> Result<Self> {
        let item_embedding = embedding(item_vocab_size, embed_dim, vb.pp("item_embedding"))?;

        let other_features: Vec<FeatureSpec> = features
            .iter()
            .filter(|feature| feature.name != behavior_feature && feature.name != candidate_feature)
            .cloned()
            .collect();
        let embeddings = if other_features.is_empty() {
            None
        } else {
            Some(FeatureEmbeddings::new(
                vb.pp("embeddings"),
                &other_features,
            )?)
        };
        let other_dim = embeddings
            .as_ref()
            .map_or(0, |embeddings| embeddings.total_dim);

        let activation_unit = Mlp::new(
            vb.pp("activation_unit"),
            4 * embed_dim,
            activation_hidden_dims,
            1,
            Activation::Relu,
        )?;

        let mlp = Mlp::new(
            vb.pp("mlp"),
            embed_dim + embed_dim + other_dim,
            mlp_hidden_dims,
            1,
            Activation::Relu,
        )?;

        Ok(Self {
            item_embedding,
            embeddings,
            activation_unit,
            mlp,
            output_head: None,
            embed_dim,
            behavior_feature: behavior_feature.to_string(),
            candidate_feature: candidate_feature.to_string(),
        })
    }

    /// Construct DIN with output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        item_vocab_size: usize,
        embed_dim: usize,
        activation_hidden_dims: &[usize],
        mlp_hidden_dims: &[usize],
        behavior_feature: &str,
        candidate_feature: &str,
        contract: &OutputContract,
    ) -> Result<Self> {
        let mut model = Self::new(
            vb.clone(),
            features,
            item_vocab_size,
            embed_dim,
            activation_hidden_dims,
            mlp_hidden_dims,
            behavior_feature,
            candidate_feature,
        )?;
        let representation_dims = HashMap::from([("shared".to_string(), 1)]);
        model.output_head = Some(OutputHead::new(
            contract,
            &representation_dims,
            vb.pp("output_head"),
        )?);
        Ok(model)
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let behavior_indices = x_inputs.get(&self.behavior_feature).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Missing behavior feature '{}'",
                self.behavior_feature
            ))
        })?;
        let candidate_indices = x_inputs.get(&self.candidate_feature).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Missing candidate feature '{}'",
                self.candidate_feature
            ))
        })?;

        let behavior_indices = if behavior_indices.rank() == 1 {
            behavior_indices.unsqueeze(1)?
        } else {
            behavior_indices.clone()
        };
        let behavior_embs = self.item_embedding.forward(&behavior_indices)?;
        let candidate_emb = self.item_embedding.forward(candidate_indices)?;

        let seq_len = behavior_embs.dim(1)?;
        let embed_dim = self.embed_dim;
        let batch = behavior_embs.dim(0)?;

        let candidate_emb_3d = candidate_emb
            .unsqueeze(1)?
            .broadcast_as(behavior_embs.shape())?;
        let prod = behavior_embs.mul(&candidate_emb_3d)?;
        let diff = behavior_embs.sub(&candidate_emb_3d)?;

        let au_input = Tensor::cat(&[&behavior_embs, &candidate_emb_3d, &prod, &diff], 2)?;
        let au_input_flat = au_input.reshape((batch * seq_len, 4 * embed_dim))?;
        let attn_weights = self
            .activation_unit
            .forward(&au_input_flat)?
            .reshape((batch, seq_len, 1))?;

        let interest_emb = behavior_embs
            .mul(&attn_weights.broadcast_as(behavior_embs.shape())?)?
            .sum(1)?;

        match &self.embeddings {
            Some(embeddings) => {
                let other_emb = embeddings.forward(x_inputs)?;
                let combined = Tensor::cat(&[&interest_emb, &candidate_emb, &other_emb], 1)?;
                self.mlp.forward(&combined)
            }
            None => {
                let combined = Tensor::cat(&[&interest_emb, &candidate_emb], 1)?;
                self.mlp.forward(&combined)
            }
        }
    }
}

impl Model for DIN {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let logits = self.shared(x_inputs)?;
        let mut outputs = ModelOutput::new();
        outputs.insert_binary_logit("pred", logits);
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
