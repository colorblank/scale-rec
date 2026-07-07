//! DCN V2: Improved Deep & Cross Network (arXiv:2008.13535).
//!
//! Single-task CTR model with gated cross network + optional deep MLP.
//! Cross network uses the DCNv2-style gated cross layer (sigmoid-gated feature crossing).

use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::gdcn::GatedCrossNetwork;
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use std::collections::HashMap;

/// DCN V2 model: gated cross network + optional deep MLP.
pub struct DCNV2 {
    embeddings: FeatureEmbeddings,
    cross: GatedCrossNetwork,
    deep: Option<Mlp>,
    shared_bottom: Option<Mlp>,
    output_head: Option<OutputHead>,
}

impl DCNV2 {
    /// Construct DCNV2 with cross network + optional deep + optional shared bottom.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        cross_layers: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let input_dim = embeddings.total_dim;
        let cross = GatedCrossNetwork::new(vb.pp("cross"), input_dim, cross_layers)?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, input_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("deep"),
                input_dim,
                &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), input_dim + output_dim)
        };

        let shared_bottom = if shared_bottom_dims.is_empty() {
            None
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            Some(Mlp::new(
                vb.pp("shared_bottom"),
                fusion_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?)
        };

        Ok(Self {
            embeddings,
            cross,
            deep,
            shared_bottom,
            output_head: None,
        })
    }

    /// Construct DCNV2 with output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        cross_layers: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        contract: &OutputContract,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let input_dim = embeddings.total_dim;
        let cross = GatedCrossNetwork::new(vb.pp("cross"), input_dim, cross_layers)?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, input_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("deep"),
                    input_dim,
                    &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                input_dim + output_dim,
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
            cross,
            deep,
            shared_bottom,
            output_head: Some(output_head),
        })
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let dense = self.embeddings.forward(x_inputs)?;
        let cross_out = self.cross.forward(&dense)?;
        let mut shared = match &self.deep {
            Some(deep) => Tensor::cat(&[cross_out, deep.forward(&dense)?], 1)?,
            None => cross_out,
        };
        if let Some(shared_bottom) = &self.shared_bottom {
            shared = shared_bottom.forward(&shared)?;
        }
        Ok(shared)
    }
}

impl Model for DCNV2 {
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
