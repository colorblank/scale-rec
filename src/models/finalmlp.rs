//! FinalMLP: Two-stream MLP with feature gating and interaction aggregation.
//!
//! Two parallel MLPs receive differently gated feature inputs;
//! their outputs are fused via learnable bilinear interaction aggregation.
//! Paper: arXiv:2304.00902 (AAAI 2023).

use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::{linear, linear_no_bias, Init, Linear, Module, VarBuilder};
use std::collections::HashMap;

/// FinalMLP model: two-stream MLP with feature gating and bilinear aggregation.
pub struct FinalMLP {
    embeddings: FeatureEmbeddings,
    stream1_gate_0: Linear,
    stream1_gate_1: Linear,
    stream2_gate_0: Linear,
    stream2_gate_1: Linear,
    stream1_mlp: Mlp,
    stream2_mlp: Mlp,
    fusion_o1: Linear,
    fusion_o2: Linear,
    fusion_bilinear: Linear,
    fusion_bias: Tensor,
    output_head: Option<OutputHead>,
}

impl FinalMLP {
    #[allow(clippy::too_many_arguments)]
    /// Construct FinalMLP: shared embeddings → two gated streams → bilinear fusion.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        stream_hidden_dims: &[usize],
        gate_hidden_dim: usize,
        _fusion_hidden_dims: &[usize],
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;

        let stream_output_dim = *stream_hidden_dims.last().unwrap_or(&1);

        let stream1_gate_0 = linear(total_dim, gate_hidden_dim, vb.pp("stream1_gate.0"))?;
        let stream1_gate_1 = linear(gate_hidden_dim, total_dim, vb.pp("stream1_gate.1"))?;
        let stream2_gate_0 = linear(total_dim, gate_hidden_dim, vb.pp("stream2_gate.0"))?;
        let stream2_gate_1 = linear(gate_hidden_dim, total_dim, vb.pp("stream2_gate.1"))?;

        let stream1_mlp = Mlp::new(
            vb.pp("stream1_mlp"),
            total_dim,
            &stream_hidden_dims[..stream_hidden_dims.len().saturating_sub(1)],
            stream_output_dim,
            Activation::Relu,
        )?;

        let stream2_mlp = Mlp::new(
            vb.pp("stream2_mlp"),
            total_dim,
            &stream_hidden_dims[..stream_hidden_dims.len().saturating_sub(1)],
            stream_output_dim,
            Activation::Relu,
        )?;

        let fusion_o1 = linear_no_bias(stream_output_dim, 1, vb.pp("fusion_o1"))?;
        let fusion_o2 = linear_no_bias(stream_output_dim, 1, vb.pp("fusion_o2"))?;
        let fusion_bilinear = linear_no_bias(
            stream_output_dim,
            stream_output_dim,
            vb.pp("fusion_bilinear"),
        )?;
        let fusion_bias = vb.get_with_hints((1,), "fusion_bias", Init::Const(0.0))?;

        Ok(Self {
            embeddings,
            stream1_gate_0,
            stream1_gate_1,
            stream2_gate_0,
            stream2_gate_1,
            stream1_mlp,
            stream2_mlp,
            fusion_o1,
            fusion_o2,
            fusion_bilinear,
            fusion_bias,
            output_head: None,
        })
    }

    /// Construct FinalMLP with output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        stream_hidden_dims: &[usize],
        gate_hidden_dim: usize,
        fusion_hidden_dims: &[usize],
        contract: &OutputContract,
    ) -> Result<Self> {
        let mut model = Self::new(
            vb.clone(),
            features,
            stream_hidden_dims,
            gate_hidden_dim,
            fusion_hidden_dims,
        )?;
        let representation_dims = HashMap::from([("shared".to_string(), 1)]);
        model.output_head = Some(OutputHead::new(
            contract,
            &representation_dims,
            vb.pp("output_head"),
        )?);
        Ok(model)
    }

    fn compute_gate(x: &Tensor, linear0: &Linear, linear1: &Linear) -> Result<Tensor> {
        let hidden = linear0.forward(x)?.relu()?;
        candle_nn::ops::sigmoid(&linear1.forward(&hidden)?)?.affine(2.0, 0.0)
    }

    fn bilinear_fusion(&self, h1: &Tensor, h2: &Tensor) -> Result<Tensor> {
        let linear_terms = self
            .fusion_o1
            .forward(h1)?
            .broadcast_add(&self.fusion_o2.forward(h2)?)?;
        let bilinear = self.fusion_bilinear.forward(h1)?.mul(h2)?.sum_keepdim(1)?;
        linear_terms
            .broadcast_add(&bilinear)?
            .broadcast_add(&self.fusion_bias)
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(x_inputs)?;

        let gate1 = Self::compute_gate(&x, &self.stream1_gate_0, &self.stream1_gate_1)?;
        let x1 = x.mul(&gate1)?;
        let h1 = self.stream1_mlp.forward(&x1)?;

        let gate2 = Self::compute_gate(&x, &self.stream2_gate_0, &self.stream2_gate_1)?;
        let x2 = x.mul(&gate2)?;
        let h2 = self.stream2_mlp.forward(&x2)?;

        self.bilinear_fusion(&h1, &h2)
    }
}

impl Model for FinalMLP {
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
