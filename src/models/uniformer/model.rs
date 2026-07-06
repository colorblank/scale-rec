//! UniFormer: feature interaction module + task interaction module.

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, linear, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use crate::feats::config::PoolingStrategy;
use crate::layers::embedding::FeatureSpec;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::{Model, ModelExecution, ModelOutput};

/// UniFormer construction config.
#[derive(Debug, Clone)]
pub struct UniFormerConfig {
    /// Hidden token dimension.
    pub d: usize,
    /// FFN hidden dimension.
    pub d_ff: usize,
    /// Number of FIM/TIM layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of task tokens exposed as `task_i` representations.
    pub num_tasks: usize,
}

/// UniFormer model.
pub struct UniFormerModel {
    tokenizer: UniFormerTokenizer,
    fim_layers: Vec<FeatureInteractionLayer>,
    tim_layers: Vec<TaskInteractionLayer>,
    task_tokens: Tensor,
    final_norm: RmsNorm,
    task_towers: Option<MultiTaskTower>,
    output_head: Option<OutputHead>,
}

impl UniFormerModel {
    /// Construct with legacy task towers.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: UniFormerConfig,
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let mut model = Self::build(vb.clone(), features, config)?;
        model.task_towers = Some(MultiTaskTower::new(
            task_config,
            model.hidden_dim(),
            vb.pp("task_towers"),
        )?);
        Ok(model)
    }

    /// Construct with native output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: UniFormerConfig,
        contract: &OutputContract,
    ) -> Result<Self> {
        let mut model = Self::build(vb.clone(), features, config)?;
        let mut representation_dims = HashMap::from([("shared".to_string(), model.hidden_dim())]);
        for task_idx in 0..model.num_tasks() {
            representation_dims.insert(format!("task_{task_idx}"), model.hidden_dim());
        }
        model.output_head = Some(OutputHead::new(
            contract,
            &representation_dims,
            vb.pp("output_head"),
        )?);
        Ok(model)
    }

    fn build(vb: VarBuilder, features: &[FeatureSpec], config: UniFormerConfig) -> Result<Self> {
        validate_config(&config)?;
        let tokenizer = UniFormerTokenizer::new(vb.pp("tokenizer"), features, config.d)?;
        let mut fim_layers = Vec::with_capacity(config.num_layers);
        let mut tim_layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            fim_layers.push(FeatureInteractionLayer::new(
                vb.pp("fim_layers").pp(layer_idx.to_string()),
                config.d,
                config.d_ff,
                config.n_heads,
            )?);
            tim_layers.push(TaskInteractionLayer::new(
                vb.pp("tim_layers").pp(layer_idx.to_string()),
                config.d,
                config.d_ff,
                config.n_heads,
            )?);
        }
        let task_tokens = vb.get_with_hints(
            (config.num_tasks, config.d),
            "task_tokens",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let final_norm = rms_norm(config.d, 1e-5, vb.pp("final_norm"))?;
        Ok(Self {
            tokenizer,
            fim_layers,
            tim_layers,
            task_tokens,
            final_norm,
            task_towers: None,
            output_head: None,
        })
    }

    fn hidden_dim(&self) -> usize {
        self.tokenizer.d
    }

    fn num_tasks(&self) -> usize {
        self.task_tokens.dims()[0]
    }

    fn representations(
        &self,
        x_inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let encoded = self.tokenizer.forward(x_inputs)?;
        let mut feature_tokens = encoded.non_sequence_tokens;
        let sequence_tokens = encoded.sequence_tokens;
        for layer in &self.fim_layers {
            feature_tokens = layer.forward(&feature_tokens, &sequence_tokens)?;
        }
        let batch = feature_tokens.dim(0)?;
        let mut task_tokens = self.task_tokens.unsqueeze(0)?.broadcast_as((
            batch,
            self.num_tasks(),
            self.hidden_dim(),
        ))?;
        for layer in &self.tim_layers {
            task_tokens = layer.forward(&task_tokens, &feature_tokens)?;
        }
        task_tokens = self.final_norm.forward(&task_tokens)?;
        let mut reps = HashMap::from([("shared".to_string(), task_tokens.mean(1)?)]);
        for task_idx in 0..self.num_tasks() {
            reps.insert(
                format!("task_{task_idx}"),
                task_tokens.narrow(1, task_idx, 1)?.squeeze(1)?,
            );
        }
        Ok(reps)
    }
}

impl Model for UniFormerModel {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let reps = self.representations(x_inputs)?;
        self.task_towers
            .as_ref()
            .unwrap()
            .forward(reps.get("shared").unwrap())
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let reps = self.representations(x_inputs)?;
        if let Some(head) = &self.output_head {
            return head.forward(&reps);
        }
        let outputs = self
            .task_towers
            .as_ref()
            .unwrap()
            .forward(reps.get("shared").unwrap())?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}

fn validate_config(config: &UniFormerConfig) -> Result<()> {
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
    if config.num_tasks == 0 {
        candle_core::bail!("num_tasks must be > 0");
    }
    Ok(())
}

struct EncodedFeatures {
    non_sequence_tokens: Tensor,
    sequence_tokens: Tensor,
}

struct UniFormerTokenizer {
    feature_to_idx: HashMap<String, usize>,
    ordered_names: Vec<String>,
    specs: Vec<FeatureSpec>,
    embeddings: Vec<Embedding>,
    sequence_projections: Vec<Linear>,
    non_sequence_projections: Vec<Linear>,
    d: usize,
}

impl UniFormerTokenizer {
    fn new(vb: VarBuilder, features: &[FeatureSpec], d: usize) -> Result<Self> {
        if features.is_empty() {
            candle_core::bail!("UniFormer requires at least one feature");
        }
        let mut feature_to_idx = HashMap::with_capacity(features.len());
        let mut ordered_names = Vec::with_capacity(features.len());
        let mut specs = Vec::with_capacity(features.len());
        let mut embeddings = Vec::with_capacity(features.len());
        let mut sequence_projections = Vec::with_capacity(features.len());
        let mut non_sequence_projections = Vec::with_capacity(features.len());
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
        }
        Ok(Self {
            feature_to_idx,
            ordered_names,
            specs,
            embeddings,
            sequence_projections,
            non_sequence_projections,
            d,
        })
    }

    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<EncodedFeatures> {
        let mut non_sequence_tokens = Vec::new();
        let mut sequence_tokens = Vec::new();
        for name in &self.ordered_names {
            let input = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            let spec = &self.specs[idx];
            let emb = self.embeddings[idx].forward(input)?;
            if spec.seq_len.is_some() && emb.rank() == 3 {
                sequence_tokens.push(self.sequence_projections[idx].forward(&emb)?);
            } else {
                let pooled = pool_feature(spec, emb)?;
                non_sequence_tokens.push(
                    self.non_sequence_projections[idx]
                        .forward(&pooled)?
                        .unsqueeze(1)?,
                );
            }
        }
        if sequence_tokens.is_empty() {
            sequence_tokens.extend(non_sequence_tokens.iter().cloned());
        }
        if non_sequence_tokens.is_empty() {
            let sequence_summary = Tensor::cat(&sequence_tokens, 1)?.mean(1)?.unsqueeze(1)?;
            non_sequence_tokens.push(sequence_summary);
        }
        Ok(EncodedFeatures {
            non_sequence_tokens: Tensor::cat(&non_sequence_tokens, 1)?,
            sequence_tokens: Tensor::cat(&sequence_tokens, 1)?,
        })
    }
}

struct FeatureInteractionLayer {
    cross_attn: MultiHeadAttention,
    self_attn: MultiHeadAttention,
    ffn: SwiGluFfn,
    norm_cross: RmsNorm,
    norm_self: RmsNorm,
    norm_ffn: RmsNorm,
}

impl FeatureInteractionLayer {
    fn new(vb: VarBuilder, d: usize, d_ff: usize, n_heads: usize) -> Result<Self> {
        Ok(Self {
            cross_attn: MultiHeadAttention::new(vb.pp("cross_attn"), d, n_heads)?,
            self_attn: MultiHeadAttention::new(vb.pp("self_attn"), d, n_heads)?,
            ffn: SwiGluFfn::new(vb.pp("ffn"), d, d_ff)?,
            norm_cross: rms_norm(d, 1e-5, vb.pp("norm_cross"))?,
            norm_self: rms_norm(d, 1e-5, vb.pp("norm_self"))?,
            norm_ffn: rms_norm(d, 1e-5, vb.pp("norm_ffn"))?,
        })
    }

    fn forward(&self, non_sequence: &Tensor, sequence: &Tensor) -> Result<Tensor> {
        let x = non_sequence.broadcast_add(&self.cross_attn.forward(
            &self.norm_cross.forward(non_sequence)?,
            sequence,
            sequence,
        )?)?;
        let y = x.broadcast_add(&self.self_attn.forward(
            &self.norm_self.forward(&x)?,
            &x,
            &x,
        )?)?;
        y.broadcast_add(&self.ffn.forward(&self.norm_ffn.forward(&y)?)?)
    }
}

struct TaskInteractionLayer {
    cross_attn: MultiHeadAttention,
    self_attn: MultiHeadAttention,
    ffn: SwiGluFfn,
    norm_cross: RmsNorm,
    norm_self: RmsNorm,
    norm_ffn: RmsNorm,
}

impl TaskInteractionLayer {
    fn new(vb: VarBuilder, d: usize, d_ff: usize, n_heads: usize) -> Result<Self> {
        Ok(Self {
            cross_attn: MultiHeadAttention::new(vb.pp("cross_attn"), d, n_heads)?,
            self_attn: MultiHeadAttention::new(vb.pp("self_attn"), d, n_heads)?,
            ffn: SwiGluFfn::new(vb.pp("ffn"), d, d_ff)?,
            norm_cross: rms_norm(d, 1e-5, vb.pp("norm_cross"))?,
            norm_self: rms_norm(d, 1e-5, vb.pp("norm_self"))?,
            norm_ffn: rms_norm(d, 1e-5, vb.pp("norm_ffn"))?,
        })
    }

    fn forward(&self, task_tokens: &Tensor, feature_tokens: &Tensor) -> Result<Tensor> {
        let x = task_tokens.broadcast_add(&self.cross_attn.forward(
            &self.norm_cross.forward(task_tokens)?,
            feature_tokens,
            feature_tokens,
        )?)?;
        let y = x.broadcast_add(&self.self_attn.forward(
            &self.norm_self.forward(&x)?,
            &x,
            &x,
        )?)?;
        y.broadcast_add(&self.ffn.forward(&self.norm_ffn.forward(&y)?)?)
    }
}

struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    d: usize,
    n_heads: usize,
    d_head: usize,
}

impl MultiHeadAttention {
    fn new(vb: VarBuilder, d: usize, n_heads: usize) -> Result<Self> {
        Ok(Self {
            q_proj: linear(d, d, vb.pp("q_proj"))?,
            k_proj: linear(d, d, vb.pp("k_proj"))?,
            v_proj: linear(d, d, vb.pp("v_proj"))?,
            o_proj: linear(d, d, vb.pp("o_proj"))?,
            d,
            n_heads,
            d_head: d / n_heads,
        })
    }

    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let (batch, q_len, _) = query.dims3()?;
        let k_len = key.dim(1)?;
        let q = self
            .q_proj
            .forward(query)?
            .reshape((batch, q_len, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(key)?
            .reshape((batch, k_len, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(value)?
            .reshape((batch, k_len, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let scale = (self.d_head as f32).sqrt();
        let kt = k.permute((0, 1, 3, 2))?.contiguous()?;
        let attn = candle_nn::ops::softmax(
            &q.matmul(&kt)?
                .broadcast_div(&Tensor::from_slice(&[scale], (1,), query.device())?)?,
            3,
        )?;
        self.o_proj.forward(
            &attn
                .matmul(&v)?
                .permute((0, 2, 1, 3))?
                .contiguous()?
                .reshape((batch, q_len, self.d))?,
        )
    }
}

struct SwiGluFfn {
    up: Linear,
    gate: Linear,
    down: Linear,
}

impl SwiGluFfn {
    fn new(vb: VarBuilder, d: usize, d_ff: usize) -> Result<Self> {
        Ok(Self {
            up: linear(d, d_ff, vb.pp("up"))?,
            gate: linear(d, d_ff, vb.pp("gate"))?,
            down: linear(d_ff, d, vb.pp("down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(
            &self
                .up
                .forward(x)?
                .gelu()?
                .mul(&candle_nn::ops::sigmoid(&self.gate.forward(x)?)?)?,
        )
    }
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
