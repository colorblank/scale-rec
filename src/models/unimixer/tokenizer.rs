//! FeatureTokenizer：分组 Conv1d 将离散特征投影为 Token 序列。
use candle_core::{Result, Tensor};
use candle_nn::{conv1d, embedding, Conv1d, Conv1dConfig, Embedding, Module, VarBuilder};
use std::collections::HashMap;

pub struct FeatureTokenizer {
    feature_to_emb_idx: HashMap<String, usize>,
    ordered_feature_names: Vec<String>,
    embeddings: Vec<Embedding>,
    token_projections: Conv1d,
    pub num_tokens: usize,
    pub token_dim: usize,
}

impl FeatureTokenizer {
    pub fn new(
        vb: VarBuilder,
        features: &[(String, usize, usize)],
        token_dim: usize,
        num_tokens: usize,
    ) -> Result<Self> {
        let mut feature_to_emb_idx = HashMap::with_capacity(features.len());
        let mut ordered_feature_names = Vec::with_capacity(features.len());
        let mut embeddings = Vec::with_capacity(features.len());
        let mut total_embed_dim = 0;
        for (i, (name, vocab_size, embed_dim)) in features.iter().enumerate() {
            feature_to_emb_idx.insert(name.clone(), i);
            ordered_feature_names.push(name.clone());
            let emb = embedding(*vocab_size, *embed_dim, vb.pp(format!("emb_{}", name)))?;
            embeddings.push(emb);
            total_embed_dim += embed_dim;
        }
        if total_embed_dim % num_tokens != 0 {
            candle_core::bail!(
                "Total embed dim ({}) must be divisible by num_tokens ({})",
                total_embed_dim,
                num_tokens
            );
        }
        let conv_config = Conv1dConfig {
            groups: num_tokens,
            ..Default::default()
        };
        let token_projections = conv1d(
            total_embed_dim,
            num_tokens * token_dim,
            1,
            conv_config,
            vb.pp("token_projections"),
        )?;
        Ok(Self {
            feature_to_emb_idx,
            ordered_feature_names,
            embeddings,
            token_projections,
            token_dim,
            num_tokens,
        })
    }

    pub fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut embeds = Vec::with_capacity(self.ordered_feature_names.len());
        for name in &self.ordered_feature_names {
            let input_tensor = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let emb_idx = *self.feature_to_emb_idx.get(name).unwrap();
            let emb_out = self.embeddings[emb_idx].forward(input_tensor)?;
            embeds.push(emb_out);
        }
        let concat_embeds = Tensor::cat(&embeds, 1)?;
        let batch_size = concat_embeds.dim(0)?;
        let conv_in = concat_embeds.unsqueeze(2)?;
        let conv_out = self.token_projections.forward(&conv_in)?;
        let squeezed = conv_out.squeeze(2)?;
        let output_tokens = squeezed.reshape((batch_size, self.num_tokens, self.token_dim))?;
        Ok(output_tokens)
    }
}
