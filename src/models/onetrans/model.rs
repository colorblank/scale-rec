//! OneTrans model.
//!
//! Implements the repo-facing parts of OneTrans (arXiv:2510.26104): unified
//! sequence/non-sequence tokenization, causal transformer blocks, shared
//! parameterization for sequence tokens, token-specific parameters for
//! non-sequence tokens, and optional pyramid tail truncation.

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::feats::config::PoolingStrategy;
use crate::layers::embedding::FeatureSpec;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::{Model, ModelExecution, ModelOutput};

/// OneTrans construction config.
#[derive(Debug, Clone)]
pub struct OneTransConfig {
    /// Hidden token dimension.
    pub d: usize,
    /// FFN hidden dimension.
    pub d_ff: usize,
    /// Number of causal transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Optional number of tail tokens kept after each layer.
    pub pyramid_tail_tokens: Option<usize>,
}

/// OneTrans model.
pub struct OneTransModel {
    tokenizer: OneTransTokenizer,
    blocks: Vec<OneTransBlock>,
    final_norm: RmsNorm,
    output_proj: Linear,
    task_towers: Option<MultiTaskTower>,
    output_head: Option<OutputHead>,
}

impl OneTransModel {
    /// Construct with legacy task towers.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: OneTransConfig,
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let tokenizer = OneTransTokenizer::new(vb.pp("tokenizer"), features, config.d)?;
        let blocks = build_blocks(vb.pp("blocks"), &config, tokenizer.max_tokens)?;
        let final_norm = rms_norm(config.d, 1e-5, vb.pp("final_norm"))?;
        let output_proj = linear(config.d, config.d, vb.pp("output_proj"))?;
        let task_towers = MultiTaskTower::new(task_config, config.d, vb.pp("task_towers"))?;
        Ok(Self {
            tokenizer,
            blocks,
            final_norm,
            output_proj,
            task_towers: Some(task_towers),
            output_head: None,
        })
    }

    /// Construct with native output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: OneTransConfig,
        contract: &OutputContract,
    ) -> Result<Self> {
        let tokenizer = OneTransTokenizer::new(vb.pp("tokenizer"), features, config.d)?;
        let blocks = build_blocks(vb.pp("blocks"), &config, tokenizer.max_tokens)?;
        let final_norm = rms_norm(config.d, 1e-5, vb.pp("final_norm"))?;
        let output_proj = linear(config.d, config.d, vb.pp("output_proj"))?;
        let output_head = OutputHead::new(
            contract,
            &HashMap::from([("shared".to_string(), config.d)]),
            vb.pp("output_head"),
        )?;
        Ok(Self {
            tokenizer,
            blocks,
            final_norm,
            output_proj,
            task_towers: None,
            output_head: Some(output_head),
        })
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let mut encoded = self.tokenizer.forward(x_inputs)?;
        for block in &self.blocks {
            encoded = block.forward(&encoded)?;
        }
        let x = self.final_norm.forward(&encoded.tokens)?;
        self.output_proj.forward(&x.mean(1)?)
    }
}

impl Model for OneTransModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        self.task_towers
            .as_ref()
            .unwrap()
            .forward(&self.shared(x_inputs)?)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let shared = self.shared(x_inputs)?;
        if let Some(head) = &self.output_head {
            return head.forward(&HashMap::from([("shared".to_string(), shared)]));
        }
        let outputs = self.task_towers.as_ref().unwrap().forward(&shared)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}

fn build_blocks(
    vb: VarBuilder,
    config: &OneTransConfig,
    max_tokens: usize,
) -> Result<Vec<OneTransBlock>> {
    if config.d == 0 {
        candle_core::bail!("d must be > 0");
    }
    if config.d_ff == 0 {
        candle_core::bail!("d_ff must be > 0");
    }
    if config.num_layers == 0 {
        candle_core::bail!("num_layers must be > 0");
    }
    if config.n_heads == 0 || config.d % config.n_heads != 0 {
        candle_core::bail!(
            "d ({}) must be divisible by n_heads ({})",
            config.d,
            config.n_heads
        );
    }
    if matches!(config.pyramid_tail_tokens, Some(0)) {
        candle_core::bail!("pyramid_tail_tokens must be > 0 when set");
    }
    let mut blocks = Vec::with_capacity(config.num_layers);
    for idx in 0..config.num_layers {
        blocks.push(OneTransBlock::new(
            vb.pp(idx.to_string()),
            config.d,
            config.d_ff,
            config.n_heads,
            max_tokens,
            config.pyramid_tail_tokens,
        )?);
    }
    Ok(blocks)
}

struct EncodedTokens {
    tokens: Tensor,
    is_sequence: Vec<bool>,
}

struct OneTransTokenizer {
    feature_to_idx: HashMap<String, usize>,
    ordered_names: Vec<String>,
    specs: Vec<FeatureSpec>,
    embeddings: Vec<Embedding>,
    sequence_projections: Vec<Linear>,
    non_sequence_projections: Vec<Linear>,
    position_embedding: Tensor,
    max_tokens: usize,
}

impl OneTransTokenizer {
    fn new(vb: VarBuilder, features: &[FeatureSpec], d: usize) -> Result<Self> {
        if features.is_empty() {
            candle_core::bail!("OneTrans requires at least one feature");
        }
        if d == 0 {
            candle_core::bail!("d must be > 0");
        }
        let mut feature_to_idx = HashMap::with_capacity(features.len());
        let mut ordered_names = Vec::with_capacity(features.len());
        let mut specs = Vec::with_capacity(features.len());
        let mut embeddings = Vec::with_capacity(features.len());
        let mut sequence_projections = Vec::with_capacity(features.len());
        let mut non_sequence_projections = Vec::with_capacity(features.len());
        let mut max_tokens = 0;
        for (idx, spec) in features.iter().enumerate() {
            feature_to_idx.insert(spec.name.clone(), idx);
            ordered_names.push(spec.name.clone());
            specs.push(spec.clone());
            embeddings.push(embedding(
                spec.vocab_size,
                spec.embed_dim,
                vb.pp("embeddings").pp(format!("emb_{}", spec.name)),
            )?);
            sequence_projections.push(linear(
                spec.embed_dim,
                d,
                vb.pp("sequence_projections").pp(idx.to_string()),
            )?);
            non_sequence_projections.push(linear(
                feature_output_dim(spec)?,
                d,
                vb.pp("non_sequence_projections").pp(idx.to_string()),
            )?);
            max_tokens += spec.seq_len.unwrap_or(1);
        }
        let position_embedding = vb.get_with_hints(
            (max_tokens, d),
            "position_embedding",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        Ok(Self {
            feature_to_idx,
            ordered_names,
            specs,
            embeddings,
            sequence_projections,
            non_sequence_projections,
            position_embedding,
            max_tokens,
        })
    }

    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<EncodedTokens> {
        let mut sequence_tokens = Vec::new();
        let mut non_sequence_tokens = Vec::new();
        let mut is_sequence = Vec::new();
        for name in &self.ordered_names {
            let input = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            let spec = &self.specs[idx];
            let emb = self.embeddings[idx].forward(input)?;
            if spec.seq_len.is_some() && emb.rank() == 3 {
                let projected = self.sequence_projections[idx].forward(&emb)?;
                is_sequence.extend(std::iter::repeat(true).take(projected.dim(1)?));
                sequence_tokens.push(projected);
            } else {
                let pooled = pool_feature(spec, emb)?;
                let token = self.non_sequence_projections[idx]
                    .forward(&pooled)?
                    .unsqueeze(1)?;
                is_sequence.push(false);
                non_sequence_tokens.push(token);
            }
        }
        let mut parts = sequence_tokens;
        parts.extend(non_sequence_tokens);
        let tokens = Tensor::cat(&parts, 1)?;
        let n = tokens.dim(1)?;
        let pos = self.position_embedding.narrow(0, 0, n)?.unsqueeze(0)?;
        Ok(EncodedTokens {
            tokens: tokens.broadcast_add(&pos)?,
            is_sequence,
        })
    }
}

struct OneTransBlock {
    norm1: RmsNorm,
    q_shared: Linear,
    k_shared: Linear,
    v_shared: Linear,
    q_non_sequence: Linear,
    k_non_sequence: Linear,
    v_non_sequence: Linear,
    o_proj: Linear,
    norm2: RmsNorm,
    seq_ffn_up: Linear,
    seq_ffn_down: Linear,
    ns_ffn_up: Linear,
    ns_ffn_down: Linear,
    d: usize,
    n_heads: usize,
    d_head: usize,
    pyramid_tail_tokens: Option<usize>,
}

impl OneTransBlock {
    fn new(
        vb: VarBuilder,
        d: usize,
        d_ff: usize,
        n_heads: usize,
        _max_tokens: usize,
        pyramid_tail_tokens: Option<usize>,
    ) -> Result<Self> {
        let norm1 = rms_norm(d, 1e-5, vb.pp("norm1"))?;
        let q_shared = linear(d, d, vb.pp("attn.q_shared"))?;
        let k_shared = linear(d, d, vb.pp("attn.k_shared"))?;
        let v_shared = linear(d, d, vb.pp("attn.v_shared"))?;
        let q_non_sequence = linear(d, d, vb.pp("attn.q_non_sequence"))?;
        let k_non_sequence = linear(d, d, vb.pp("attn.k_non_sequence"))?;
        let v_non_sequence = linear(d, d, vb.pp("attn.v_non_sequence"))?;
        let ns_ffn_up = linear(d, d_ff, vb.pp("ns_ffn.up"))?;
        let ns_ffn_down = linear(d_ff, d, vb.pp("ns_ffn.down"))?;
        Ok(Self {
            norm1,
            q_shared,
            k_shared,
            v_shared,
            q_non_sequence,
            k_non_sequence,
            v_non_sequence,
            o_proj: linear(d, d, vb.pp("attn.o_proj"))?,
            norm2: rms_norm(d, 1e-5, vb.pp("norm2"))?,
            seq_ffn_up: linear(d, d_ff, vb.pp("seq_ffn.up"))?,
            seq_ffn_down: linear(d_ff, d, vb.pp("seq_ffn.down"))?,
            ns_ffn_up,
            ns_ffn_down,
            d,
            n_heads,
            d_head: d / n_heads,
            pyramid_tail_tokens,
        })
    }

    fn forward(&self, encoded: &EncodedTokens) -> Result<EncodedTokens> {
        let residual = &encoded.tokens;
        let x = self.norm1.forward(residual)?;
        let q = self.project_mixed(
            &x,
            &encoded.is_sequence,
            &self.q_shared,
            &self.q_non_sequence,
        )?;
        let k = self.project_mixed(
            &x,
            &encoded.is_sequence,
            &self.k_shared,
            &self.k_non_sequence,
        )?;
        let v = self.project_mixed(
            &x,
            &encoded.is_sequence,
            &self.v_shared,
            &self.v_non_sequence,
        )?;
        let (b, n, _) = q.dims3()?;
        let q = q
            .reshape((b, n, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = k
            .reshape((b, n, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = v
            .reshape((b, n, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let scale = (self.d_head as f32).sqrt();
        let mask = causal_mask(n, x.device())?;
        let kt = k.permute((0, 1, 3, 2))?.contiguous()?;
        let scores = q
            .matmul(&kt)?
            .broadcast_div(&Tensor::from_slice(&[scale], (1,), x.device())?)?
            .broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let x = attn
            .matmul(&v)?
            .permute((0, 2, 1, 3))?
            .reshape((b, n, self.d))?;
        let x = residual.broadcast_add(&self.o_proj.forward(&x)?)?;

        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let x = self.forward_ffn(&x, &encoded.is_sequence)?;
        let tokens = residual.broadcast_add(&x)?;
        self.truncate_tail(tokens, &encoded.is_sequence)
    }

    fn project_mixed(
        &self,
        x: &Tensor,
        is_sequence: &[bool],
        shared: &Linear,
        non_sequence: &Linear,
    ) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(is_sequence.len());
        for (idx, is_seq) in is_sequence.iter().copied().enumerate() {
            let token = x.narrow(1, idx, 1)?.squeeze(1)?;
            let projected = if is_seq {
                shared.forward(&token)?
            } else {
                non_sequence.forward(&token)?
            };
            outputs.push(projected.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }

    fn forward_ffn(&self, x: &Tensor, is_sequence: &[bool]) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(is_sequence.len());
        for (idx, is_seq) in is_sequence.iter().copied().enumerate() {
            let token = x.narrow(1, idx, 1)?.squeeze(1)?;
            let projected = if is_seq {
                self.seq_ffn_down
                    .forward(&self.seq_ffn_up.forward(&token)?.gelu()?)?
            } else {
                self.ns_ffn_down
                    .forward(&self.ns_ffn_up.forward(&token)?.gelu()?)?
            };
            outputs.push(projected.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }

    fn truncate_tail(&self, tokens: Tensor, is_sequence: &[bool]) -> Result<EncodedTokens> {
        let Some(keep) = self.pyramid_tail_tokens else {
            return Ok(EncodedTokens {
                tokens,
                is_sequence: is_sequence.to_vec(),
            });
        };
        let n = tokens.dim(1)?;
        if n <= keep {
            return Ok(EncodedTokens {
                tokens,
                is_sequence: is_sequence.to_vec(),
            });
        }
        Ok(EncodedTokens {
            tokens: tokens.narrow(1, n - keep, keep)?,
            is_sequence: is_sequence[n - keep..].to_vec(),
        })
    }
}

fn causal_mask(n: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut data = vec![0f32; n * n];
    for row in 0..n {
        for col in row + 1..n {
            data[row * n + col] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(&data, (n, n), device)
}

fn pool_feature(spec: &FeatureSpec, emb: Tensor) -> Result<Tensor> {
    if emb.rank() != 3 {
        return Ok(emb);
    }
    match spec.pooling {
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
