//! FeatureTokenizer：分组 Conv1d 将离散特征投影为 Token 序列。
use super::profile;
use candle_core::{Result, Tensor};
use candle_nn::{conv1d, embedding, Conv1d, Conv1dConfig, Embedding, Linear, Module, VarBuilder};
use std::collections::HashMap;

use crate::feats::config::PoolingStrategy;
use crate::layers::embedding::FeatureSpec;

/// 特征分词器：分组 Conv1d 将多特征投影为 Token 序列。
pub struct FeatureTokenizer {
    feature_to_emb_idx: HashMap<String, usize>,
    ordered_feature_names: Vec<String>,
    feature_specs: Vec<FeatureSpec>,
    embeddings: Vec<Embedding>,
    token_projection_linears: Vec<Linear>,
    token_input_dim: usize,
    /// 输出 token 数量。
    pub num_tokens: usize,
    /// 每个 token 的维度。
    pub token_dim: usize,
}

impl FeatureTokenizer {
    /// 构造 FeatureTokenizer。
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        token_dim: usize,
        num_tokens: usize,
    ) -> Result<Self> {
        if num_tokens == 0 {
            candle_core::bail!("num_tokens must be > 0");
        }
        if token_dim == 0 {
            candle_core::bail!("token_dim must be > 0");
        }
        let mut feature_to_emb_idx = HashMap::with_capacity(features.len());
        let mut ordered_feature_names = Vec::with_capacity(features.len());
        let mut feature_specs = Vec::with_capacity(features.len());
        let mut embeddings = Vec::with_capacity(features.len());
        let mut total_embed_dim = 0;
        for (i, spec) in features.iter().enumerate() {
            feature_to_emb_idx.insert(spec.name.clone(), i);
            ordered_feature_names.push(spec.name.clone());
            feature_specs.push(spec.clone());
            let emb = embedding(
                spec.vocab_size,
                spec.embed_dim,
                vb.pp(format!("emb_{}", spec.name)),
            )?;
            embeddings.push(emb);
            total_embed_dim += feature_output_dim(spec)?;
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
        let token_input_dim = total_embed_dim / num_tokens;
        let token_projection_linears =
            build_token_projection_linears(&token_projections, num_tokens, token_dim)?;
        Ok(Self {
            feature_to_emb_idx,
            ordered_feature_names,
            feature_specs,
            embeddings,
            token_projection_linears,
            token_input_dim,
            token_dim,
            num_tokens,
        })
    }

    fn pool(&self, idx: usize, emb: Tensor) -> Result<Tensor> {
        if emb.rank() != 3 {
            return Ok(emb);
        }
        match self.feature_specs[idx].pooling {
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

    /// 前向：特征嵌入 → 分组投影 → [batch, num_tokens, token_dim]。
    pub fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let total_timer = profile::start();
        let embedding_timer = profile::start();
        let mut embeds = Vec::with_capacity(self.ordered_feature_names.len());
        for name in &self.ordered_feature_names {
            let feature_timer = profile::verbose().then(std::time::Instant::now);
            let input_tensor = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let emb_idx = *self.feature_to_emb_idx.get(name).unwrap();
            let emb_out = self.embeddings[emb_idx].forward(input_tensor)?;
            embeds.push(self.pool(emb_idx, emb_out)?);
            profile::log(&format!("tokenizer.feature.{name}"), feature_timer);
        }
        profile::log("tokenizer.embedding_pool", embedding_timer);
        let cat_timer = profile::start();
        let concat_embeds = Tensor::cat(&embeds, 1)?;
        profile::log("tokenizer.concat_embeds", cat_timer);
        let projection_timer = profile::start();
        let batch_size = concat_embeds.dim(0)?;
        let token_inputs =
            concat_embeds.reshape((batch_size, self.num_tokens, self.token_input_dim))?;
        let mut outputs = Vec::with_capacity(self.num_tokens);
        for token_idx in 0..self.num_tokens {
            let token_input = token_inputs
                .narrow(1, token_idx, 1)?
                .squeeze(1)?
                .contiguous()?;
            outputs.push(
                self.token_projection_linears[token_idx]
                    .forward(&token_input)?
                    .unsqueeze(1)?,
            );
        }
        let output = Tensor::cat(&outputs, 1)?;
        profile::log("tokenizer.token_projection", projection_timer);
        profile::log("tokenizer.total", total_timer);
        Ok(output)
    }
}

fn build_token_projection_linears(
    token_projections: &Conv1d,
    num_tokens: usize,
    token_dim: usize,
) -> Result<Vec<Linear>> {
    let weight = token_projections.weight();
    let bias = token_projections.bias();
    let mut linears = Vec::with_capacity(num_tokens);
    for token_idx in 0..num_tokens {
        let offset = token_idx * token_dim;
        let token_weight = weight.narrow(0, offset, token_dim)?.squeeze(2)?;
        let token_bias = match bias {
            Some(bias) => Some(bias.narrow(0, offset, token_dim)?),
            None => None,
        };
        linears.push(Linear::new(token_weight, token_bias));
    }
    Ok(linears)
}

fn feature_output_dim(spec: &FeatureSpec) -> Result<usize> {
    match (spec.pooling, spec.seq_len) {
        (PoolingStrategy::Flatten, Some(seq_len)) if seq_len > 0 => Ok(spec.embed_dim * seq_len),
        (PoolingStrategy::Flatten, _) => {
            candle_core::bail!(
                "feature '{}' pooling flatten requires seq_len > 0",
                spec.name
            )
        }
        _ => Ok(spec.embed_dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn linear_token_projection_matches_grouped_conv1d() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let num_tokens = 2;
        let token_dim = 3;
        let total_embed_dim = 4;
        let token_input_dim = total_embed_dim / num_tokens;
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
        )
        .unwrap();
        let linears =
            build_token_projection_linears(&token_projections, num_tokens, token_dim).unwrap();
        let concat_embeds = Tensor::from_slice(
            &[
                0.1f32, -0.2, 0.3, -0.4, //
                1.0, 2.0, -1.0, -2.0,
            ],
            (2, total_embed_dim),
            &device,
        )
        .unwrap();

        let conv_out = token_projections
            .forward(&concat_embeds.unsqueeze(2).unwrap())
            .unwrap()
            .squeeze(2)
            .unwrap()
            .reshape((2, num_tokens, token_dim))
            .unwrap();

        let token_inputs = concat_embeds
            .reshape((2, num_tokens, token_input_dim))
            .unwrap();
        let mut linear_outputs = Vec::with_capacity(num_tokens);
        for token_idx in 0..num_tokens {
            let token_input = token_inputs
                .narrow(1, token_idx, 1)
                .unwrap()
                .squeeze(1)
                .unwrap()
                .contiguous()
                .unwrap();
            linear_outputs.push(
                linears[token_idx]
                    .forward(&token_input)
                    .unwrap()
                    .unsqueeze(1)
                    .unwrap(),
            );
        }
        let linear_out = Tensor::cat(&linear_outputs, 1).unwrap();

        let diff = linear_out
            .sub(&conv_out)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff <= 1e-6, "max abs diff too large: {diff}");
    }
}
