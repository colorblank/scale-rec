//! HyFormer: hybrid query decoding plus query boosting for CTR prediction.

use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};

use crate::feats::config::PoolingStrategy;
use crate::layers::embedding::FeatureSpec;
use crate::layers::towers::{MultiTaskConfig, MultiTaskTower};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use crate::models::rankmixer::block::RankMixerBlock;
use crate::models::{Model, ModelExecution, ModelOutput};

/// HyFormer construction config.
#[derive(Debug, Clone)]
pub struct HyFormerConfig {
    /// Hidden token dimension.
    pub d: usize,
    /// Query FFN hidden dimension.
    pub d_ff: usize,
    /// Number of generated global query tokens.
    pub num_queries: usize,
    /// Number of HyFormer layers.
    pub num_layers: usize,
    /// Per-token FFN hidden multiplier in query boosting.
    pub hidden_factor: f64,
}

/// HyFormer model.
pub struct HyFormerModel {
    tokenizer: HyFormerTokenizer,
    query_generator: QueryGenerator,
    layers: Vec<HyFormerLayer>,
    output_proj: Linear,
    task_towers: Option<MultiTaskTower>,
    output_head: Option<OutputHead>,
}

impl HyFormerModel {
    /// Construct with legacy task towers.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: HyFormerConfig,
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let tokenizer = HyFormerTokenizer::new(vb.pp("tokenizer"), features, config.d)?;
        let query_generator = QueryGenerator::new(
            vb.pp("query_generator"),
            tokenizer.global_input_dim,
            config.d,
            config.d_ff,
            config.num_queries,
        )?;
        let layers = build_layers(
            vb.pp("layers"),
            tokenizer.boost_token_count(config.num_queries),
            &config,
        )?;
        let output_proj = linear(config.d, config.d, vb.pp("output_proj"))?;
        let task_towers = MultiTaskTower::new(task_config, config.d, vb.pp("task_towers"))?;
        Ok(Self {
            tokenizer,
            query_generator,
            layers,
            output_proj,
            task_towers: Some(task_towers),
            output_head: None,
        })
    }

    /// Construct with native output contract.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        config: HyFormerConfig,
        contract: &OutputContract,
    ) -> Result<Self> {
        let tokenizer = HyFormerTokenizer::new(vb.pp("tokenizer"), features, config.d)?;
        let query_generator = QueryGenerator::new(
            vb.pp("query_generator"),
            tokenizer.global_input_dim,
            config.d,
            config.d_ff,
            config.num_queries,
        )?;
        let layers = build_layers(
            vb.pp("layers"),
            tokenizer.boost_token_count(config.num_queries),
            &config,
        )?;
        let output_proj = linear(config.d, config.d, vb.pp("output_proj"))?;
        let output_head = OutputHead::new(
            contract,
            &HashMap::from([("shared".to_string(), config.d)]),
            vb.pp("output_head"),
        )?;
        Ok(Self {
            tokenizer,
            query_generator,
            layers,
            output_proj,
            task_towers: None,
            output_head: Some(output_head),
        })
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let encoded = self.tokenizer.forward(x_inputs)?;
        let mut queries = self.query_generator.forward(&encoded.global_info)?;
        for layer in &self.layers {
            queries = layer.forward(&queries, &encoded.memory, &encoded.non_sequence_tokens)?;
        }
        self.output_proj.forward(&queries.mean(1)?)
    }
}

impl Model for HyFormerModel {
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

struct EncodedFeatures {
    global_info: Tensor,
    non_sequence_tokens: Tensor,
    memory: Tensor,
}

struct HyFormerTokenizer {
    feature_to_idx: HashMap<String, usize>,
    ordered_names: Vec<String>,
    specs: Vec<FeatureSpec>,
    embeddings: Vec<Embedding>,
    pooled_projections: Vec<Linear>,
    sequence_projections: Vec<Linear>,
    non_sequence_count: usize,
    global_input_dim: usize,
}

impl HyFormerTokenizer {
    fn new(vb: VarBuilder, features: &[FeatureSpec], d: usize) -> Result<Self> {
        if features.is_empty() {
            candle_core::bail!("HyFormer requires at least one feature");
        }
        if d == 0 {
            candle_core::bail!("d must be > 0");
        }
        let mut feature_to_idx = HashMap::with_capacity(features.len());
        let mut ordered_names = Vec::with_capacity(features.len());
        let mut specs = Vec::with_capacity(features.len());
        let mut embeddings = Vec::with_capacity(features.len());
        let mut pooled_projections = Vec::with_capacity(features.len());
        let mut sequence_projections = Vec::with_capacity(features.len());
        let mut non_sequence_count = 0;
        let mut global_input_dim = 0;
        for (idx, spec) in features.iter().enumerate() {
            feature_to_idx.insert(spec.name.clone(), idx);
            ordered_names.push(spec.name.clone());
            specs.push(spec.clone());
            embeddings.push(embedding(
                spec.vocab_size,
                spec.embed_dim,
                vb.pp("embeddings").pp(format!("emb_{}", spec.name)),
            )?);
            pooled_projections.push(linear(
                spec.output_dim(),
                d,
                vb.pp("pooled_projections").pp(idx.to_string()),
            )?);
            sequence_projections.push(linear(
                spec.embed_dim,
                d,
                vb.pp("sequence_projections").pp(idx.to_string()),
            )?);
            if spec.seq_len.is_none() {
                non_sequence_count += 1;
            }
            global_input_dim += spec.output_dim();
        }
        Ok(Self {
            feature_to_idx,
            ordered_names,
            specs,
            embeddings,
            pooled_projections,
            sequence_projections,
            non_sequence_count,
            global_input_dim,
        })
    }

    fn boost_token_count(&self, num_queries: usize) -> usize {
        num_queries + self.non_sequence_count.max(1)
    }

    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<EncodedFeatures> {
        let mut global_parts = Vec::with_capacity(self.ordered_names.len());
        let mut ns_tokens = Vec::with_capacity(self.non_sequence_count.max(1));
        let mut memory_tokens = Vec::new();
        for name in &self.ordered_names {
            let input = x_inputs
                .get(name)
                .ok_or_else(|| candle_core::Error::Msg(format!("Feature '{}' not found", name)))?;
            let idx = *self.feature_to_idx.get(name).unwrap();
            let spec = &self.specs[idx];
            let emb = self.embeddings[idx].forward(input)?;
            let pooled = pool_feature(spec, emb.clone())?;
            global_parts.push(pooled.clone());
            if spec.seq_len.is_some() && emb.rank() == 3 {
                memory_tokens.push(project_sequence(&self.sequence_projections[idx], &emb)?);
            } else {
                let token = self.pooled_projections[idx]
                    .forward(&pooled)?
                    .unsqueeze(1)?;
                ns_tokens.push(token.clone());
                memory_tokens.push(token);
            }
        }
        if ns_tokens.is_empty() {
            ns_tokens.push(memory_tokens[0].mean(1)?.unsqueeze(1)?);
        }
        Ok(EncodedFeatures {
            global_info: Tensor::cat(&global_parts, 1)?,
            non_sequence_tokens: Tensor::cat(&ns_tokens, 1)?,
            memory: Tensor::cat(&memory_tokens, 1)?,
        })
    }
}

struct QueryGenerator {
    up: Linear,
    down: Linear,
    num_queries: usize,
    d: usize,
}

impl QueryGenerator {
    fn new(
        vb: VarBuilder,
        input_dim: usize,
        d: usize,
        d_ff: usize,
        num_queries: usize,
    ) -> Result<Self> {
        if num_queries == 0 {
            candle_core::bail!("num_queries must be > 0");
        }
        let hidden = d_ff.max(1);
        Ok(Self {
            up: linear(input_dim, hidden, vb.pp("up"))?,
            down: linear(hidden, num_queries * d, vb.pp("down"))?,
            num_queries,
            d,
        })
    }

    fn forward(&self, global_info: &Tensor) -> Result<Tensor> {
        let hidden = self.up.forward(global_info)?.gelu()?;
        let out = self.down.forward(&hidden)?;
        let batch = out.dim(0)?;
        out.reshape((batch, self.num_queries, self.d))
    }
}

struct HyFormerLayer {
    norm_query: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    boost: RankMixerBlock,
    num_queries: usize,
}

impl HyFormerLayer {
    fn new(
        vb: VarBuilder,
        d: usize,
        token_count: usize,
        num_queries: usize,
        hidden_factor: f64,
    ) -> Result<Self> {
        Ok(Self {
            norm_query: layer_norm(d, 1e-5, vb.pp("norm_query"))?,
            q_proj: linear(d, d, vb.pp("q_proj"))?,
            k_proj: linear(d, d, vb.pp("k_proj"))?,
            v_proj: linear(d, d, vb.pp("v_proj"))?,
            out_proj: linear(d, d, vb.pp("out_proj"))?,
            boost: RankMixerBlock::new(d, token_count, token_count, hidden_factor, vb.pp("boost"))?,
            num_queries,
        })
    }

    fn forward(&self, queries: &Tensor, memory: &Tensor, ns_tokens: &Tensor) -> Result<Tensor> {
        let decoded = self.decode(queries, memory)?;
        let mixed_input = Tensor::cat(&[&decoded, ns_tokens], 1)?;
        let boosted = self.boost.forward(&mixed_input)?;
        boosted.narrow(1, 0, self.num_queries)
    }

    fn decode(&self, queries: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(&self.norm_query.forward(queries)?)?;
        let k = self.k_proj.forward(memory)?;
        let v = self.v_proj.forward(memory)?;
        let scores = q.matmul(&k.permute((0, 2, 1))?)?;
        let scale = (q.dim(2)? as f32).sqrt();
        let scores = scores.broadcast_div(&Tensor::from_slice(&[scale], (1,), q.device())?)?;
        let attn = candle_nn::ops::softmax(&scores, 2)?;
        let decoded = attn.matmul(&v)?;
        let projected = self.out_proj.forward(&decoded)?;
        projected.add(queries)
    }
}

fn build_layers(
    vb: VarBuilder,
    token_count: usize,
    config: &HyFormerConfig,
) -> Result<Vec<HyFormerLayer>> {
    if config.num_layers == 0 {
        candle_core::bail!("num_layers must be > 0");
    }
    if config.hidden_factor <= 0.0 {
        candle_core::bail!("hidden_factor must be > 0");
    }
    if config.d % token_count != 0 {
        candle_core::bail!(
            "d ({}) must be divisible by HyFormer boost token count ({})",
            config.d,
            token_count
        );
    }
    let mut layers = Vec::with_capacity(config.num_layers);
    for idx in 0..config.num_layers {
        layers.push(HyFormerLayer::new(
            vb.pp(idx.to_string()),
            config.d,
            token_count,
            config.num_queries,
            config.hidden_factor,
        )?);
    }
    Ok(layers)
}

fn project_sequence(projection: &Linear, emb: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, dim) = emb.dims3()?;
    projection
        .forward(&emb.reshape((batch * seq_len, dim))?)?
        .reshape((batch, seq_len, ()))
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
