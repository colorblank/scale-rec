//! RankUp tokenizer: shuffled sparse groups plus optional global/cross/task tokens.

use crate::feats::config::PoolingStrategy;
use crate::layers::embedding::FeatureSpec;
use candle_core::{Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};
use std::collections::HashMap;

/// Optional interaction token between two learned feature embeddings.
#[derive(Debug, Clone)]
pub struct CrossTokenConfig {
    /// Left feature name.
    pub left: String,
    /// Right feature name.
    pub right: String,
}

/// Tokenizer implementing the input-side RankUp mechanisms supported by this repo.
pub struct RankUpTokenizer {
    feature_to_idx: HashMap<String, usize>,
    ordered_feature_names: Vec<String>,
    feature_specs: Vec<FeatureSpec>,
    embeddings: Vec<Embedding>,
    extra_embeddings: Vec<Vec<Embedding>>,
    sparse_groups: Vec<Vec<usize>>,
    token_projections: Vec<Linear>,
    global_projection: Option<Linear>,
    cross_projection: Option<Linear>,
    cross_pair: Option<(usize, usize)>,
    task_tokens: Option<Tensor>,
    /// Output token count including auxiliary tokens.
    pub num_tokens: usize,
    /// Output token dimension.
    pub token_dim: usize,
}

impl RankUpTokenizer {
    /// Construct a RankUp tokenizer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        token_dim: usize,
        num_sparse_tokens: usize,
        permutation_seed: u64,
        multi_embedding_tables: usize,
        use_global_token: bool,
        cross_token: Option<CrossTokenConfig>,
        num_task_tokens: usize,
    ) -> Result<Self> {
        if features.is_empty() {
            candle_core::bail!("RankUp requires at least one feature");
        }
        if token_dim == 0 {
            candle_core::bail!("token_dim must be > 0");
        }
        if num_sparse_tokens == 0 {
            candle_core::bail!("num_sparse_tokens must be > 0");
        }
        if num_sparse_tokens > features.len() {
            candle_core::bail!(
                "num_sparse_tokens ({}) cannot exceed feature count ({})",
                num_sparse_tokens,
                features.len()
            );
        }
        if multi_embedding_tables == 0 {
            candle_core::bail!("multi_embedding_tables must be > 0");
        }

        let mut feature_to_idx = HashMap::with_capacity(features.len());
        let mut ordered_feature_names = Vec::with_capacity(features.len());
        let mut feature_specs = Vec::with_capacity(features.len());
        let mut embeddings = Vec::with_capacity(features.len());
        let mut extra_embeddings = Vec::with_capacity(multi_embedding_tables.saturating_sub(1));
        for table_idx in 1..multi_embedding_tables {
            let mut table = Vec::with_capacity(features.len());
            for spec in features {
                table.push(embedding(
                    spec.vocab_size,
                    spec.embed_dim,
                    vb.pp(format!("multi_emb_{}_{}", table_idx, spec.name)),
                )?);
            }
            extra_embeddings.push(table);
        }

        let mut feature_dims = Vec::with_capacity(features.len());
        for (i, spec) in features.iter().enumerate() {
            feature_to_idx.insert(spec.name.clone(), i);
            ordered_feature_names.push(spec.name.clone());
            feature_specs.push(spec.clone());
            embeddings.push(embedding(
                spec.vocab_size,
                spec.embed_dim,
                vb.pp(format!("emb_{}", spec.name)),
            )?);
            feature_dims.push(feature_output_dim(spec)? * multi_embedding_tables);
        }

        let sparse_groups = shuffled_groups(features.len(), num_sparse_tokens, permutation_seed);
        let mut token_projections = Vec::with_capacity(num_sparse_tokens);
        for (group_idx, group) in sparse_groups.iter().enumerate() {
            let input_dim: usize = group.iter().map(|idx| feature_dims[*idx]).sum();
            token_projections.push(linear(
                input_dim,
                token_dim,
                vb.pp(format!("token_projections.{}", group_idx)),
            )?);
        }

        let total_dim: usize = feature_dims.iter().sum();
        let global_projection = if use_global_token {
            Some(linear(total_dim, token_dim, vb.pp("global_projection"))?)
        } else {
            None
        };

        let (cross_pair, cross_projection) = if let Some(pair) = cross_token {
            let left = *feature_to_idx.get(&pair.left).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "cross token left feature '{}' not found",
                    pair.left
                ))
            })?;
            let right = *feature_to_idx.get(&pair.right).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "cross token right feature '{}' not found",
                    pair.right
                ))
            })?;
            let left_dim = feature_output_dim(&features[left])?;
            let right_dim = feature_output_dim(&features[right])?;
            if left_dim != right_dim {
                candle_core::bail!(
                    "cross token features '{}' and '{}' must have equal pooled dims, got {} and {}",
                    pair.left,
                    pair.right,
                    left_dim,
                    right_dim
                );
            }
            (
                Some((left, right)),
                Some(linear(left_dim, token_dim, vb.pp("cross_projection"))?),
            )
        } else {
            (None, None)
        };

        let task_tokens = if num_task_tokens > 0 {
            Some(vb.get_with_hints(
                (num_task_tokens, token_dim),
                "task_tokens",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?)
        } else {
            None
        };
        let num_tokens = num_sparse_tokens
            + usize::from(use_global_token)
            + usize::from(cross_pair.is_some())
            + num_task_tokens;

        Ok(Self {
            feature_to_idx,
            ordered_feature_names,
            feature_specs,
            embeddings,
            extra_embeddings,
            sparse_groups,
            token_projections,
            global_projection,
            cross_projection,
            cross_pair,
            task_tokens,
            num_tokens,
            token_dim,
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

    /// Forward to `[batch, num_tokens, token_dim]`.
    pub fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut feature_embeds = Vec::with_capacity(self.ordered_feature_names.len());
        for name in &self.ordered_feature_names {
            let input_tensor = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            let mut parts = Vec::with_capacity(1 + self.extra_embeddings.len());
            parts.push(self.pool(idx, self.embeddings[idx].forward(input_tensor)?)?);
            for table in &self.extra_embeddings {
                parts.push(self.pool(idx, table[idx].forward(input_tensor)?)?);
            }
            feature_embeds.push(Tensor::cat(&parts, 1)?);
        }

        let mut tokens = Vec::with_capacity(self.num_tokens);
        for (group_idx, group) in self.sparse_groups.iter().enumerate() {
            let parts: Vec<&Tensor> = group.iter().map(|idx| &feature_embeds[*idx]).collect();
            let grouped = Tensor::cat(&parts, 1)?;
            tokens.push(
                self.token_projections[group_idx]
                    .forward(&grouped)?
                    .unsqueeze(1)?,
            );
        }
        if let Some(global_projection) = &self.global_projection {
            let parts: Vec<&Tensor> = feature_embeds.iter().collect();
            tokens.push(
                global_projection
                    .forward(&Tensor::cat(&parts, 1)?)?
                    .unsqueeze(1)?,
            );
        }
        if let (Some((left, right)), Some(cross_projection)) =
            (self.cross_pair, &self.cross_projection)
        {
            let cross = feature_embeds[left].mul(&feature_embeds[right])?;
            tokens.push(cross_projection.forward(&cross)?.unsqueeze(1)?);
        }
        if let Some(task_tokens) = &self.task_tokens {
            let batch = feature_embeds[0].dim(0)?;
            let task_count = task_tokens.dim(0)?;
            tokens.push(
                task_tokens
                    .unsqueeze(0)?
                    .expand((batch, task_count, self.token_dim))?,
            );
        }
        Tensor::cat(&tokens, 1)
    }

    /// Number of appended task tokens.
    pub fn num_task_tokens(&self) -> usize {
        self.task_tokens
            .as_ref()
            .map_or(0, |tokens| tokens.dim(0).unwrap_or(0))
    }
}

fn shuffled_groups(count: usize, groups: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut indices: Vec<usize> = (0..count).collect();
    let mut state = seed.max(1);
    for i in (1..indices.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
    let mut out = vec![Vec::new(); groups];
    for (i, idx) in indices.into_iter().enumerate() {
        out[i % groups].push(idx);
    }
    out
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
