use candle_core::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::collections::HashMap;

pub struct FeatureEmbeddings {
    feature_to_idx: HashMap<String, usize>,
    ordered_names: Vec<String>,
    embeddings: Vec<Embedding>,
    pub num_features: usize,
    pub total_dim: usize,
}

impl FeatureEmbeddings {
    pub fn new(vb: VarBuilder, features: &[(String, usize, usize)]) -> Result<Self> {
        let num_features = features.len();
        let mut feature_to_idx = HashMap::with_capacity(num_features);
        let mut ordered_names = Vec::with_capacity(num_features);
        let mut embeddings = Vec::with_capacity(num_features);
        let mut total_dim = 0;
        for (i, (name, vocab_size, embed_dim)) in features.iter().enumerate() {
            feature_to_idx.insert(name.clone(), i);
            ordered_names.push(name.clone());
            embeddings.push(embedding(*vocab_size, *embed_dim, vb.pp(format!("emb_{}", name)))?);
            total_dim += embed_dim;
        }
        Ok(Self { feature_to_idx, ordered_names, embeddings, num_features, total_dim })
    }

    pub fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut embeds = Vec::with_capacity(self.num_features);
        for name in &self.ordered_names {
            let input_tensor = x_inputs.get(name).ok_or_else(|| {
                candle_core::Error::Msg(format!("Feature '{}' not found", name))
            })?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            embeds.push(self.embeddings[idx].forward(input_tensor)?);
        }
        Tensor::cat(&embeds, 1)
    }

    pub fn forward_stacked(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let mut embeds = Vec::with_capacity(self.num_features);
        for name in &self.ordered_names {
            let input_tensor = x_inputs.get(name).ok_or_else(|| {
                candle_core::Error::Msg(format!("Feature '{}' not found", name))
            })?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            embeds.push(self.embeddings[idx].forward(input_tensor)?.unsqueeze(1)?);
        }
        Ok(embeds)
    }
}
