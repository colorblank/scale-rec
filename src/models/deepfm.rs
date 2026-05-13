use super::Model;
use crate::layers::embedding::FeatureEmbeddings;
use crate::layers::fm::fm_interaction;
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

pub struct DeepFM {
    fm_first_embeddings: FeatureEmbeddings,
    fm_second_embeddings: FeatureEmbeddings,
    pub fm_k: usize,
    deep_embeddings: FeatureEmbeddings,
    pub deep_total_dim: usize,
    deep_mlp: Mlp,
    global_bias: Tensor,
}

impl DeepFM {
    pub fn new(
        vb: VarBuilder,
        features: &[(String, usize, usize)],
        fm_k: usize,
        deep_hidden_dims: &[usize],
    ) -> Result<Self> {
        let fm_first_cfg: Vec<(String, usize, usize)> = features
            .iter()
            .map(|(n, v, _)| (n.clone(), *v, 1))
            .collect();
        let fm_first_embeddings = FeatureEmbeddings::new(vb.pp("fm_first"), &fm_first_cfg)?;
        let fm_second_cfg: Vec<(String, usize, usize)> = features
            .iter()
            .map(|(n, v, _)| (n.clone(), *v, fm_k))
            .collect();
        let fm_second_embeddings = FeatureEmbeddings::new(vb.pp("fm_second"), &fm_second_cfg)?;
        let deep_embeddings = FeatureEmbeddings::new(vb.pp("deep"), features)?;
        let deep_mlp = Mlp::new(
            vb.pp("deep_mlp"),
            deep_embeddings.total_dim,
            deep_hidden_dims,
            1,
            Activation::Relu,
        )?;
        let global_bias = vb.get_with_hints((1,), "global_bias", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            fm_first_embeddings,
            fm_second_embeddings,
            fm_k,
            deep_embeddings,
            deep_total_dim: features.iter().map(|(_, _, d)| d).sum(),
            deep_mlp,
            global_bias,
        })
    }
}

impl Model for DeepFM {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let first_order = self.fm_first_embeddings.forward(x_inputs)?.sum_keepdim(1)?;
        let fm_stacked = Tensor::cat(&self.fm_second_embeddings.forward_stacked(x_inputs)?, 1)?;
        let second_order = fm_interaction(&fm_stacked)?;
        let deep_input = self.deep_embeddings.forward(x_inputs)?;
        let deep_out = self.deep_mlp.forward(&deep_input)?;
        let logits = first_order
            .broadcast_add(&second_order)?
            .broadcast_add(&deep_out)?
            .broadcast_add(&self.global_bias)?;
        let mut outputs = HashMap::new();
        outputs.insert("pred".to_string(), logits);
        Ok(outputs)
    }
}
