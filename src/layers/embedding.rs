use candle_core::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use std::collections::HashMap;

use crate::feats::config::{PoolingStrategy, TruncationSide};

#[derive(Debug, Clone)]
pub struct FeatureSpec {
    pub name: String,
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub pooling: PoolingStrategy,
    pub seq_len: Option<usize>,
    pub truncation: TruncationSide,
}

impl FeatureSpec {
    pub fn new(name: String, vocab_size: usize, embed_dim: usize) -> Self {
        Self {
            name,
            vocab_size,
            embed_dim,
            pooling: PoolingStrategy::First,
            seq_len: None,
            truncation: TruncationSide::Head,
        }
    }

    pub fn with_dim(&self, embed_dim: usize) -> Self {
        Self {
            name: self.name.clone(),
            vocab_size: self.vocab_size,
            embed_dim,
            pooling: self.pooling,
            seq_len: self.seq_len,
            truncation: self.truncation,
        }
    }

    fn output_dim(&self) -> usize {
        match (self.pooling, self.seq_len) {
            (PoolingStrategy::Flatten, Some(seq_len)) => self.embed_dim * seq_len,
            _ => self.embed_dim,
        }
    }
}

pub struct FeatureEmbeddings {
    feature_to_idx: HashMap<String, usize>,
    ordered_names: Vec<String>,
    pooling: HashMap<String, PoolingStrategy>,
    embeddings: Vec<Embedding>,
    pub num_features: usize,
    pub total_dim: usize,
}

impl FeatureEmbeddings {
    pub fn new(vb: VarBuilder, features: &[FeatureSpec]) -> Result<Self> {
        let num_features = features.len();
        let mut feature_to_idx = HashMap::with_capacity(num_features);
        let mut ordered_names = Vec::with_capacity(num_features);
        let mut pooling = HashMap::with_capacity(num_features);
        let mut embeddings = Vec::with_capacity(num_features);
        let mut total_dim = 0;

        for (i, spec) in features.iter().enumerate() {
            feature_to_idx.insert(spec.name.clone(), i);
            ordered_names.push(spec.name.clone());
            pooling.insert(spec.name.clone(), spec.pooling);
            embeddings.push(embedding(
                spec.vocab_size,
                spec.embed_dim,
                vb.pp(format!("emb_{}", spec.name)),
            )?);
            total_dim += spec.output_dim();
        }

        Ok(Self {
            feature_to_idx,
            ordered_names,
            pooling,
            embeddings,
            num_features,
            total_dim,
        })
    }

    fn pool(&self, name: &str, emb: Tensor) -> Result<Tensor> {
        if emb.rank() != 3 {
            return Ok(emb);
        }

        match self.pooling.get(name).copied().unwrap_or_default() {
            PoolingStrategy::Mean => {
                let seq_len = emb.dim(1)? as f64;
                emb.sum(1)?.affine(1.0 / seq_len, 0.0)
            }
            PoolingStrategy::Sum => emb.sum(1),
            PoolingStrategy::Max => emb.max(1),
            PoolingStrategy::Flatten => {
                let batch = emb.dim(0)?;
                let seq_len = emb.dim(1)?;
                let dim = emb.dim(2)?;
                emb.reshape((batch, seq_len * dim))
            }
            PoolingStrategy::First => emb.narrow(1, 0, 1)?.squeeze(1),
        }
    }

    pub fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut embeds = Vec::with_capacity(self.num_features);
        for name in &self.ordered_names {
            let input_tensor = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            let emb = self.embeddings[idx].forward(input_tensor)?;
            embeds.push(self.pool(name, emb)?);
        }
        Tensor::cat(&embeds, 1)
    }

    pub fn forward_stacked(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let mut embeds = Vec::with_capacity(self.num_features);
        for name in &self.ordered_names {
            let input_tensor = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            let emb = self.embeddings[idx].forward(input_tensor)?;
            embeds.push(self.pool(name, emb)?.unsqueeze(1)?);
        }
        Ok(embeds)
    }
}
