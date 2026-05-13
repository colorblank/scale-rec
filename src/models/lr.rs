use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use crate::layers::embedding::FeatureEmbeddings;
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use super::Model;

pub struct LogisticRegression { embeddings: FeatureEmbeddings, mlp: Mlp }

impl LogisticRegression {
    pub fn new(vb: VarBuilder, features: &[(String, usize, usize)]) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let mlp = Mlp::new(vb.pp("mlp"), embeddings.total_dim, &[], 1, Activation::None_)?;
        Ok(Self { embeddings, mlp })
    }
}

impl Model for LogisticRegression {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let concat = self.embeddings.forward(x_inputs)?;
        let logits = self.mlp.forward(&concat)?;
        let mut outputs = HashMap::new();
        outputs.insert("pred".to_string(), logits);
        Ok(outputs)
    }
}
