//! 模型注册中心：Model trait、ModelConfig、工厂函数注册表。
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::layers::embedding::FeatureSpec;
use crate::layers::towers::MultiTaskConfig;
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use tracing::error;

/// DeepFM 模型。
pub mod deepfm;
/// ESMM 多任务模型。
pub mod esmm;
/// FAT: Field-Aware Transformer (KDD 2026, arXiv:2511.12081)。
pub mod fat;
/// Full-Mix / RankElastor: parameterized full token mixing with GLU P-FFNs.
pub mod full_mix;
/// GDCN + ESMM 混合模型。
pub mod gdcn_esmm;
/// HyFormer: Revisiting sequence modeling and feature interaction in CTR prediction (arXiv:2601.12681).
pub mod hyformer;
/// 逻辑回归基线模型。
pub mod lr;
/// MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders (KDD 2026, arXiv:2602.14110)。
pub mod mixformer;
/// MMoE 多门控专家混合模型。
pub mod mmoe;
/// OneRank: Unified Transformer-Native Ranking Architecture (KDD 2026, arXiv:2606.16838)。
pub mod onerank;
/// OneTrans: One Transformer Fits All Features in Recommender Systems (arXiv:2510.26104).
pub mod onetrans;
/// Versioned output-contract schema and validation.
pub mod output_contract;
/// Contract-driven task towers, relation graph and public output projection.
pub mod output_head;
/// PEPNet: Parameter and Embedding Personalized Network。
pub mod pepnet;
/// RankMixer：Token Mixing + Per-token FFN 排序模型。
pub mod rankmixer;
/// RankUp: Scaling Up Top-K Sparse Features for CTR Prediction (arXiv:2604.17878).
pub mod rankup;
/// TokenMixer-Large：Mixing & Reverting 大规模排序模型。
pub mod token_mixer_large;
/// UniFormer: unified multi-view feature and task interaction model.
pub mod uniformer;
/// UniMixer 双随机矩阵交互模型。
pub mod unimixer;

/// 模型推理 trait：所有模型实现该 trait 以统一前向接口。
pub trait Model: Send + Sync {
    /// 模型前向推理，接收特征字典返回输出字典。
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput>;

    /// Execute the complete output graph. Legacy models expose the same values as nodes and outputs.
    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let outputs = self.forward(x_inputs)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }

    /// 预热 Sinkhorn-Knopp 等缓存，可选实现。
    fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// 模型输出语义类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    /// 二分类 logit，训练使用 BCEWithLogits，serving 可转换为概率。
    BinaryLogit,
    /// 已经是概率值，serving 不应再次 sigmoid。
    Probability,
    /// 回归输出，例如播放时长、金额、完成率等连续目标。
    Regression,
    /// 排序分数或 utility，通常只要求相对顺序，不要求概率校准。
    Score,
}

/// 单个模型输出，包含 tensor 及其语义类型。
#[derive(Debug, Clone)]
pub struct OutputTensor {
    /// 输出张量。
    pub tensor: Tensor,
    /// 输出语义类型。
    pub kind: OutputKind,
}

/// 结构化模型输出，避免在裸字典中混合 logits 与 probabilities。
#[derive(Debug, Clone, Default)]
pub struct ModelOutput {
    values: HashMap<String, OutputTensor>,
}

/// Complete contract execution containing internal graph nodes and public outputs.
#[derive(Debug)]
pub struct ModelExecution {
    /// Every tower and relation node, including training-only values.
    pub nodes: ModelOutput,
    /// Stable public output projection.
    pub outputs: ModelOutput,
}

impl ModelExecution {
    /// Create a complete model execution.
    pub fn new(nodes: ModelOutput, outputs: ModelOutput) -> Self {
        Self { nodes, outputs }
    }
}

impl ModelOutput {
    /// 创建空输出。
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// 插入一个二分类 logit 输出。
    pub fn insert_binary_logit(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.insert(name, tensor, OutputKind::BinaryLogit);
    }

    /// 插入一个 probability 输出。
    pub fn insert_probability(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.insert(name, tensor, OutputKind::Probability);
    }

    /// 插入一个回归输出。
    pub fn insert_regression(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.insert(name, tensor, OutputKind::Regression);
    }

    /// 插入一个排序分数输出。
    pub fn insert_score(&mut self, name: impl Into<String>, tensor: Tensor) {
        self.insert(name, tensor, OutputKind::Score);
    }

    /// 插入指定类型的输出。
    pub fn insert(&mut self, name: impl Into<String>, tensor: Tensor, kind: OutputKind) {
        self.values
            .insert(name.into(), OutputTensor { tensor, kind });
    }

    /// 获取输出。
    pub fn get(&self, name: &str) -> Option<&OutputTensor> {
        self.values.get(name)
    }

    /// 获取输出 tensor。
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.get(name).map(|output| &output.tensor)
    }

    /// 输出数量。
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// 是否没有输出。
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 是否包含指定输出。
    pub fn contains_key(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// 遍历输出。
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OutputTensor)> {
        self.values.iter()
    }
}

/// 任务配置项：名称 + 塔隐藏层维度。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfigEntry {
    /// 任务名称。
    pub name: String,
    /// 任务塔隐藏层维度。
    pub tower_dims: Vec<usize>,
}

/// 模型配置：类型标签 + 透传的 YAML params。
/// 每个模型自行解析 params，不再集中在中央枚举。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// 模型类型标签，对应模型 registry key。
    #[serde(rename = "type")]
    pub model_type: String,
    /// 模型私有 YAML 参数。
    #[serde(default, flatten)]
    pub params: serde_yaml::Value,
}

/// 模型构建选项。
#[derive(Debug, Clone)]
pub struct ModelBuildOptions {
    /// UniMixer 子模块权重前缀。
    pub unimixer_prefix: String,
}

impl Default for ModelBuildOptions {
    fn default() -> Self {
        Self {
            unimixer_prefix: "unimixer".into(),
        }
    }
}

/// Build function signature: (vb, features, tokenizer, params) -> `Box<dyn Model>`
type BuildFn = fn(
    VarBuilder,
    &[FeatureSpec],
    Option<FeatureTokenizer>,
    &serde_yaml::Value,
    &ModelBuildOptions,
) -> Result<Box<dyn Model>>;

static REGISTRY: LazyLock<HashMap<&'static str, BuildFn>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, BuildFn> = HashMap::new();
    m.insert("lr", build_lr);
    m.insert("deepfm", build_deepfm);
    m.insert("mmoe", build_mmoe);
    m.insert("esmm", build_esmm);
    m.insert("gdcn_esmm", build_gdcn_esmm);
    m.insert("unimixer", build_unimixer);
    m.insert("token_mixer_large", build_token_mixer_large);
    m.insert("rankmixer", build_rankmixer);
    m.insert("full_mix", build_full_mix);
    m.insert("rankup", build_rankup);
    m.insert("pepnet", build_pepnet);
    m.insert("hyformer", build_hyformer);
    m.insert("fat", build_fat);
    m.insert("mixformer", build_mixformer);
    m.insert("onerank", build_onerank);
    m.insert("onetrans", build_onetrans);
    m.insert("uniformer", build_uniformer);
    m
});

impl ModelConfig {
    /// 构建模型实例（默认选项）。
    pub fn build(
        &self,
        vb: VarBuilder,
        features: &[FeatureSpec],
        tokenizer: Option<FeatureTokenizer>,
    ) -> Result<Box<dyn Model>> {
        self.build_with_options(vb, features, tokenizer, &ModelBuildOptions::default())
    }

    /// 使用自定义选项构建模型实例。
    pub fn build_with_options(
        &self,
        vb: VarBuilder,
        features: &[FeatureSpec],
        tokenizer: Option<FeatureTokenizer>,
        options: &ModelBuildOptions,
    ) -> Result<Box<dyn Model>> {
        validate_model_params(self.model_type.as_str(), &self.params)?;
        match REGISTRY.get(self.model_type.as_str()) {
            Some(build_fn) => build_fn(vb, features, tokenizer, &self.params, options),
            None => candle_core::bail!(
                "Unknown model type: {}. Registered: {:?}",
                self.model_type,
                REGISTRY.keys().collect::<Vec<_>>()
            ),
        }
    }

    /// 返回模型类型字符串。
    pub fn model_type(&self) -> &str {
        &self.model_type
    }
}

// ── per-model build functions ──

fn build_lr(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(lr::LogisticRegression::with_output_contract(
            vb, features, &contract,
        )?));
    }
    Ok(Box::new(lr::LogisticRegression::new(vb, features)?))
}

fn build_deepfm(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let fm_k = yaml_usize(params, "fm_k", 16);
    let deep_hidden_dims: Vec<usize> = yaml_usize_seq(params, "deep_hidden_dims");
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(deepfm::DeepFM::with_output_contract(
            vb,
            features,
            fm_k,
            &deep_hidden_dims,
            &contract,
        )?));
    }
    Ok(Box::new(deepfm::DeepFM::new(
        vb,
        features,
        fm_k,
        &deep_hidden_dims,
    )?))
}

fn build_mmoe(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    let num_experts = yaml_usize(params, "num_experts", 4);
    let expert_hidden_dims: Vec<usize> = yaml_usize_seq(params, "expert_hidden_dims");
    let expert_output_dim = yaml_usize(params, "expert_output_dim", 32);
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(mmoe::MMoE::with_output_contract(
            vb,
            features,
            &shared_bottom_dims,
            num_experts,
            &expert_hidden_dims,
            expert_output_dim,
            &contract,
        )?));
    }
    let task_configs = parse_task_config_entries(params);
    let task_cfgs: Vec<(String, Vec<usize>)> = task_configs
        .iter()
        .map(|t| (t.name.clone(), t.tower_dims.clone()))
        .collect();
    Ok(Box::new(mmoe::MMoE::new(
        vb,
        features,
        &shared_bottom_dims,
        num_experts,
        &expert_hidden_dims,
        expert_output_dim,
        &task_cfgs,
    )?))
}

fn build_esmm(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(esmm::ContractEsmm::new(
            vb,
            features,
            &shared_bottom_dims,
            &contract,
        )?));
    }
    let task_config = params
        .get("task_config")
        .ok_or_else(|| candle_core::Error::Msg("ESMM requires task_config".into()))?;
    let task_config = serde_yaml::from_value(task_config.clone())
        .map_err(|e| candle_core::Error::Msg(format!("parse esmm task_config: {}", e)))?;
    Ok(Box::new(esmm::ESMM::with_task_config(
        vb,
        features,
        &shared_bottom_dims,
        &task_config,
    )?))
}

fn build_gdcn_esmm(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let cross_layers = yaml_usize(params, "cross_layers", 3);
    let deep_hidden_dims: Vec<usize> = yaml_usize_seq(params, "deep_hidden_dims");
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(gdcn_esmm::GDCNESMM::with_output_contract(
            vb,
            features,
            cross_layers,
            &deep_hidden_dims,
            &shared_bottom_dims,
            &contract,
        )?));
    }
    let task_config = params
        .get("task_config")
        .ok_or_else(|| candle_core::Error::Msg("GDCNESMM requires task_config".into()))?;
    let task_config = serde_yaml::from_value(task_config.clone())
        .map_err(|e| candle_core::Error::Msg(format!("parse gdcn_esmm task_config: {}", e)))?;
    Ok(Box::new(gdcn_esmm::GDCNESMM::with_task_config(
        vb,
        features,
        cross_layers,
        &deep_hidden_dims,
        &shared_bottom_dims,
        &task_config,
    )?))
}

fn build_unimixer(
    vb: VarBuilder,
    _features: &[FeatureSpec],
    tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let tokenizer = tokenizer.ok_or_else(|| {
        candle_core::Error::Msg("UniMixer requires external FeatureTokenizer".into())
    })?;
    let token_dim = yaml_usize(params, "token_dim", 64);
    let num_tokens = yaml_usize(params, "num_tokens", 8);
    let num_blocks = yaml_usize(params, "num_blocks", 2);
    let block_size = yaml_usize_opt(params, "block_size");
    let use_lite = yaml_bool(params, "use_lite");
    let hidden_factor = yaml_f64(params, "hidden_factor", 1.0);
    let num_basis = yaml_usize(params, "num_basis", 4);
    let rank = yaml_usize(params, "rank", 16);
    let use_siamese = yaml_bool(params, "use_siamese");
    let unimixer_vb = if options.unimixer_prefix.is_empty() {
        vb
    } else {
        vb.pp(&options.unimixer_prefix)
    };
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            unimixer::model::UniMixerModel::with_output_contract(
                tokenizer,
                token_dim,
                num_tokens,
                num_blocks,
                block_size,
                use_lite,
                hidden_factor,
                num_basis,
                rank,
                &contract,
                use_siamese,
                unimixer_vb,
            )?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(unimixer::model::UniMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        num_blocks,
        block_size,
        use_lite,
        hidden_factor,
        num_basis,
        rank,
        &task_config,
        use_siamese,
        unimixer_vb,
    )?))
}

fn build_token_mixer_large(
    vb: VarBuilder,
    _features: &[FeatureSpec],
    tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let tokenizer = tokenizer.ok_or_else(|| {
        candle_core::Error::Msg("TokenMixerLarge requires external FeatureTokenizer".into())
    })?;
    let token_dim = yaml_usize(params, "token_dim", 64);
    let num_tokens = yaml_usize(params, "num_tokens", 8);
    let num_blocks = yaml_usize(params, "num_blocks", 2);
    let num_heads = yaml_usize(params, "num_heads", 8);
    let hidden_factor = yaml_f64(params, "hidden_factor", 1.0);
    let down_init_scale = yaml_f64(params, "down_init_scale", 0.01);
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            token_mixer_large::model::TokenMixerLargeModel::with_output_contract(
                tokenizer,
                token_dim,
                num_tokens,
                num_blocks,
                num_heads,
                hidden_factor,
                &contract,
                vb,
                down_init_scale,
            )?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(
        token_mixer_large::model::TokenMixerLargeModel::new(
            tokenizer,
            token_dim,
            num_tokens,
            num_blocks,
            num_heads,
            hidden_factor,
            &task_config,
            vb,
            down_init_scale,
        )?,
    ))
}

fn build_rankmixer(
    vb: VarBuilder,
    _features: &[FeatureSpec],
    tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let tokenizer = tokenizer.ok_or_else(|| {
        candle_core::Error::Msg("RankMixer requires external FeatureTokenizer".into())
    })?;
    let token_dim = yaml_usize(params, "token_dim", 64);
    let num_tokens = yaml_usize(params, "num_tokens", 8);
    let num_blocks = yaml_usize(params, "num_blocks", 2);
    let num_heads = yaml_usize(params, "num_heads", num_tokens);
    let hidden_factor = yaml_f64(params, "hidden_factor", 1.0);
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            rankmixer::model::RankMixerModel::with_output_contract(
                tokenizer,
                token_dim,
                num_tokens,
                num_blocks,
                num_heads,
                hidden_factor,
                &contract,
                vb,
            )?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(rankmixer::model::RankMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        num_blocks,
        num_heads,
        hidden_factor,
        &task_config,
        vb,
    )?))
}

fn build_full_mix(
    vb: VarBuilder,
    _features: &[FeatureSpec],
    tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let tokenizer = tokenizer.ok_or_else(|| {
        candle_core::Error::Msg("FullMix requires external FeatureTokenizer".into())
    })?;
    let token_dim = yaml_usize(params, "token_dim", 64);
    let num_tokens = yaml_usize(params, "num_tokens", 8);
    let num_blocks = yaml_usize(params, "num_blocks", 2);
    let hidden_factor = yaml_f64(params, "hidden_factor", 2.0);
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            full_mix::model::FullMixModel::with_output_contract(
                tokenizer,
                token_dim,
                num_tokens,
                num_blocks,
                hidden_factor,
                &contract,
                vb,
            )?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(full_mix::model::FullMixModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        num_blocks,
        hidden_factor,
        &task_config,
        vb,
    )?))
}

fn build_rankup(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let cross_token = match params.get("cross_token") {
        Some(value) if !value.is_null() => Some(parse_rankup_cross_token(value)?),
        _ => None,
    };
    let config = rankup::model::RankUpConfig {
        token_dim: yaml_usize(params, "token_dim", 64),
        num_sparse_tokens: yaml_usize(params, "num_sparse_tokens", 4),
        num_blocks: yaml_usize(params, "num_blocks", 2),
        num_heads: yaml_usize_opt(params, "num_heads"),
        hidden_factor: yaml_f64(params, "hidden_factor", 1.0),
        permutation_seed: yaml_usize(params, "permutation_seed", 2026) as u64,
        multi_embedding_tables: yaml_usize(params, "multi_embedding_tables", 1),
        use_global_token: params
            .get("use_global_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(true),
        cross_token,
        num_task_tokens: yaml_usize(params, "num_task_tokens", 0),
    };
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(rankup::model::RankUpModel::with_output_contract(
            vb, features, config, &contract,
        )?));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(rankup::model::RankUpModel::new(
        vb,
        features,
        config,
        &task_config,
    )?))
}

fn build_mixformer(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let d = yaml_usize(params, "d", 384);
    let d_ff = yaml_usize(params, "d_ff", 1024);
    let num_heads = yaml_usize(params, "num_heads", 16);
    let num_layers = yaml_usize(params, "num_layers", 4);
    let contract = parse_output_contract_param(params)?
        .ok_or_else(|| candle_core::Error::Msg("MixFormer requires output_contract".into()))?;
    Ok(Box::new(mixformer::model::MixFormerModel::new(
        vb, features, d, d_ff, num_heads, num_layers, &contract,
    )?))
}

fn build_hyformer(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let config = hyformer::model::HyFormerConfig {
        d: yaml_usize(params, "d", 64),
        d_ff: yaml_usize(params, "d_ff", 128),
        num_queries: yaml_usize(params, "num_queries", 2),
        num_layers: yaml_usize(params, "num_layers", 2),
        hidden_factor: yaml_f64(params, "hidden_factor", 1.0),
    };
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            hyformer::model::HyFormerModel::with_output_contract(vb, features, config, &contract)?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(hyformer::model::HyFormerModel::new(
        vb,
        features,
        config,
        &task_config,
    )?))
}

fn build_onerank(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let d = yaml_usize(params, "d", 128);
    let d_ff = yaml_usize(params, "d_ff", 512);
    let num_layers = yaml_usize(params, "num_layers", 2);
    let n_heads = yaml_usize(params, "n_heads", 8);
    let num_tasks = yaml_usize(params, "num_tasks", 3);
    let cross_task_mask = params
        .get("cross_task_mask")
        .and_then(|v| v.as_str())
        .unwrap_or("cascade")
        .to_string();
    let contract = parse_output_contract_param(params)?
        .ok_or_else(|| candle_core::Error::Msg("OneRank requires output_contract".into()))?;
    Ok(Box::new(onerank::model::OneRankModel::new(
        vb,
        features,
        d,
        d_ff,
        num_layers,
        n_heads,
        num_tasks,
        &cross_task_mask,
        &contract,
    )?))
}

fn build_onetrans(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let config = onetrans::model::OneTransConfig {
        d: yaml_usize(params, "d", 128),
        d_ff: yaml_usize(params, "d_ff", 512),
        num_layers: yaml_usize(params, "num_layers", 2),
        n_heads: yaml_usize(params, "n_heads", 8),
        pyramid_tail_tokens: yaml_usize_opt(params, "pyramid_tail_tokens"),
    };
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            onetrans::model::OneTransModel::with_output_contract(vb, features, config, &contract)?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(onetrans::model::OneTransModel::new(
        vb,
        features,
        config,
        &task_config,
    )?))
}

fn build_uniformer(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let config = uniformer::model::UniFormerConfig {
        d: yaml_usize(params, "d", 128),
        d_ff: yaml_usize(params, "d_ff", 512),
        num_layers: yaml_usize(params, "num_layers", 2),
        n_heads: yaml_usize(params, "n_heads", 8),
        num_tasks: yaml_usize(params, "num_tasks", 3),
    };
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(
            uniformer::model::UniFormerModel::with_output_contract(
                vb, features, config, &contract,
            )?,
        ));
    }
    let task_config = parse_multi_task_config(params)?;
    Ok(Box::new(uniformer::model::UniFormerModel::new(
        vb,
        features,
        config,
        &task_config,
    )?))
}

fn build_pepnet(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let prior_dim = yaml_usize(params, "prior_dim", 16);
    let deep_hidden_dims: Vec<usize> = yaml_usize_seq(params, "deep_hidden_dims");
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    let ep_prior_features = yaml_string_seq(params, "ep_prior_features");
    let pp_prior_features = yaml_string_seq(params, "pp_prior_features");
    let domains: Vec<serde_yaml::Value> = params
        .get("domains")
        .and_then(|v| v.as_sequence())
        .map(|s| s.iter().cloned().collect())
        .unwrap_or_default();
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(pepnet::PEPNet::with_output_contract(
            vb,
            features,
            prior_dim,
            &deep_hidden_dims,
            &shared_bottom_dims,
            &contract,
            &ep_prior_features,
            &pp_prior_features,
            &domains,
        )?));
    }
    let task_config = params
        .get("task_config")
        .ok_or_else(|| candle_core::Error::Msg("PEPNet requires task_config".into()))?;
    let task_config = serde_yaml::from_value(task_config.clone())
        .map_err(|e| candle_core::Error::Msg(format!("parse pepnet task_config: {}", e)))?;
    Ok(Box::new(pepnet::PEPNet::new(
        vb,
        features,
        prior_dim,
        &deep_hidden_dims,
        &shared_bottom_dims,
        &task_config,
        &ep_prior_features,
        &pp_prior_features,
        &domains,
    )?))
}

fn build_fat(
    vb: VarBuilder,
    features: &[FeatureSpec],
    _tokenizer: Option<FeatureTokenizer>,
    params: &serde_yaml::Value,
    _options: &ModelBuildOptions,
) -> Result<Box<dyn Model>> {
    let d = yaml_usize(params, "d", 128);
    let d_ff = yaml_usize(params, "d_ff", 512);
    let num_layers = yaml_usize(params, "num_layers", 2);
    let n_heads = yaml_usize(params, "n_heads", 8);
    let m = yaml_usize(params, "M", 64);
    let k = yaml_usize(params, "k", 64);
    let k_top = yaml_usize(params, "K", 3);
    let dropout = yaml_f64(params, "dropout", 0.0);
    let deep_hidden_dims: Vec<usize> = yaml_usize_seq(params, "deep_hidden_dims");
    let shared_bottom_dims: Vec<usize> = yaml_usize_seq(params, "shared_bottom_dims");
    if let Some(contract) = parse_output_contract_param(params)? {
        return Ok(Box::new(fat::model::FATModel::with_output_contract(
            vb,
            features,
            d,
            d_ff,
            num_layers,
            n_heads,
            m,
            k,
            k_top,
            &deep_hidden_dims,
            &shared_bottom_dims,
            dropout,
            &contract,
        )?));
    }
    let task_config = params
        .get("task_config")
        .ok_or_else(|| candle_core::Error::Msg("FAT requires task_config".into()))?;
    let task_config = serde_yaml::from_value(task_config.clone())
        .map_err(|e| candle_core::Error::Msg(format!("parse fat task_config: {}", e)))?;
    Ok(Box::new(fat::model::FATModel::new(
        vb,
        features,
        d,
        d_ff,
        num_layers,
        n_heads,
        m,
        k,
        k_top,
        &deep_hidden_dims,
        &shared_bottom_dims,
        dropout,
        &task_config,
    )?))
}

// ── YAML param helpers ──

fn validate_model_params(model_type: &str, params: &serde_yaml::Value) -> Result<()> {
    let (allowed, required): (&[&str], &[&str]) = match model_type {
        "lr" => (&["tasks", "label_col_map", "metrics"], &[]),
        "deepfm" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "fm_k",
                "deep_hidden_dims",
            ],
            &[],
        ),
        "mmoe" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "shared_bottom_dims",
                "num_experts",
                "expert_hidden_dims",
                "expert_output_dim",
                "task_configs",
            ],
            &[],
        ),
        "esmm" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "shared_bottom_dims",
                "task_config",
            ],
            &["task_config"],
        ),
        "gdcn_esmm" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "cross_layers",
                "deep_hidden_dims",
                "shared_bottom_dims",
                "task_config",
            ],
            &["task_config"],
        ),
        "unimixer" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "token_dim",
                "num_tokens",
                "num_blocks",
                "block_size",
                "use_lite",
                "hidden_factor",
                "num_basis",
                "rank",
                "task_config",
                "use_siamese",
            ],
            &["task_config"],
        ),
        "token_mixer_large" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "token_dim",
                "num_tokens",
                "num_blocks",
                "num_heads",
                "hidden_factor",
                "task_config",
                "down_init_scale",
            ],
            &["task_config"],
        ),
        "rankmixer" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "token_dim",
                "num_tokens",
                "num_blocks",
                "num_heads",
                "hidden_factor",
                "task_config",
            ],
            &["task_config"],
        ),
        "full_mix" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "token_dim",
                "num_tokens",
                "num_blocks",
                "hidden_factor",
                "task_config",
            ],
            &["task_config"],
        ),
        "rankup" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "token_dim",
                "num_sparse_tokens",
                "num_blocks",
                "num_heads",
                "hidden_factor",
                "permutation_seed",
                "multi_embedding_tables",
                "use_global_token",
                "cross_token",
                "num_task_tokens",
                "task_config",
            ],
            &["task_config"],
        ),
        "pepnet" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "prior_dim",
                "deep_hidden_dims",
                "shared_bottom_dims",
                "ep_prior_features",
                "pp_prior_features",
                "domains",
                "task_config",
            ],
            &["task_config"],
        ),
        "fat" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "d",
                "d_ff",
                "num_layers",
                "n_heads",
                "M",
                "k",
                "K",
                "dropout",
                "deep_hidden_dims",
                "shared_bottom_dims",
                "task_config",
            ],
            &["task_config"],
        ),
        "mixformer" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "d",
                "d_ff",
                "num_heads",
                "num_layers",
            ],
            &[],
        ),
        "hyformer" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "d",
                "d_ff",
                "num_queries",
                "num_layers",
                "hidden_factor",
                "task_config",
            ],
            &["task_config"],
        ),
        "onerank" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "d",
                "d_ff",
                "num_layers",
                "n_heads",
                "num_tasks",
                "cross_task_mask",
            ],
            &[],
        ),
        "onetrans" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "d",
                "d_ff",
                "num_layers",
                "n_heads",
                "pyramid_tail_tokens",
                "task_config",
            ],
            &["task_config"],
        ),
        "uniformer" => (
            &[
                "tasks",
                "label_col_map",
                "metrics",
                "d",
                "d_ff",
                "num_layers",
                "n_heads",
                "num_tasks",
                "task_config",
            ],
            &["task_config"],
        ),
        _ => return Ok(()),
    };
    let has_output_contract = params.get("output_contract").is_some();
    if has_output_contract {
        for legacy in ["tasks", "task_config", "label_col_map", "metrics"] {
            if params.get(legacy).is_some() {
                candle_core::bail!(
                    "output_contract cannot be combined with legacy model field '{}'",
                    legacy
                );
            }
        }
    }
    let allowed: Vec<&str> = allowed
        .iter()
        .copied()
        .chain(std::iter::once("output_contract"))
        .collect();
    let required: Vec<&str> = required
        .iter()
        .copied()
        .filter(|key| !(has_output_contract && *key == "task_config"))
        .collect();
    validate_model_param_keys(model_type, params, &allowed, &[])?;
    if let Some(raw) = params.get("output_contract") {
        let contract: output_contract::OutputContract = serde_yaml::from_value(raw.clone())
            .map_err(|error| candle_core::Error::Msg(format!("parse output_contract: {error}")))?;
        contract.validate(None).map_err(|error| {
            candle_core::Error::Msg(format!("validate output_contract: {error}"))
        })?;
    }
    for key in [
        "tasks",
        "deep_hidden_dims",
        "shared_bottom_dims",
        "expert_hidden_dims",
        "task_configs",
        "click_hidden_dims",
        "cvr_hidden_dims",
        "detail_hidden_dims",
        "stock_hidden_dims",
        "stay_hidden_dims",
        "ep_prior_features",
        "pp_prior_features",
    ] {
        expect_optional_seq(model_type, params, key)?;
    }
    for key in [
        "label_col_map",
        "metrics",
        "task_config",
        "output_contract",
        "cross_token",
    ] {
        expect_optional_mapping(model_type, params, key)?;
    }
    for key in [
        "fm_k",
        "num_experts",
        "expert_output_dim",
        "cross_layers",
        "token_dim",
        "num_tokens",
        "num_blocks",
        "block_size",
        "num_heads",
        "num_basis",
        "rank",
        "d",
        "d_ff",
        "num_layers",
        "n_heads",
        "M",
        "k",
        "K",
        "num_tasks",
        "num_heads",
        "pyramid_tail_tokens",
        "num_sparse_tokens",
        "permutation_seed",
        "multi_embedding_tables",
        "num_task_tokens",
        "num_queries",
    ] {
        expect_optional_usize(model_type, params, key)?;
    }
    for key in ["use_lite", "use_siamese", "use_global_token"] {
        expect_optional_bool(model_type, params, key)?;
    }
    expect_optional_f64(model_type, params, "hidden_factor")?;
    validate_model_param_keys(model_type, params, &allowed, &required)?;
    Ok(())
}

fn parse_output_contract_param(
    params: &serde_yaml::Value,
) -> Result<Option<output_contract::OutputContract>> {
    params
        .get("output_contract")
        .map(|raw| {
            serde_yaml::from_value(raw.clone())
                .map_err(|error| candle_core::Error::Msg(format!("parse output_contract: {error}")))
        })
        .transpose()
}

fn validate_model_param_keys(
    model_type: &str,
    params: &serde_yaml::Value,
    allowed: &[&str],
    required: &[&str],
) -> Result<()> {
    let Some(map) = params.as_mapping() else {
        if params.is_null() && required.is_empty() {
            return Ok(());
        }
        candle_core::bail!("model '{}' params must be a mapping", model_type);
    };
    for key in map.keys().filter_map(|key| key.as_str()) {
        if !allowed.contains(&key) {
            candle_core::bail!("model '{}' params has unknown field '{}'", model_type, key);
        }
    }
    for key in required {
        if !map.contains_key(serde_yaml::Value::String((*key).to_string())) {
            candle_core::bail!(
                "model '{}' params missing required field '{}'",
                model_type,
                key
            );
        }
    }
    Ok(())
}

fn model_param<'a>(params: &'a serde_yaml::Value, key: &str) -> Option<&'a serde_yaml::Value> {
    params.get(key).filter(|value| !value.is_null())
}

fn expect_optional_seq(model_type: &str, params: &serde_yaml::Value, key: &str) -> Result<()> {
    if let Some(value) = model_param(params, key) {
        if value.as_sequence().is_none() {
            candle_core::bail!("model '{}' params.{} must be list", model_type, key);
        }
    }
    Ok(())
}

fn expect_optional_mapping(model_type: &str, params: &serde_yaml::Value, key: &str) -> Result<()> {
    if let Some(value) = model_param(params, key) {
        if value.as_mapping().is_none() {
            candle_core::bail!("model '{}' params.{} must be mapping", model_type, key);
        }
    }
    Ok(())
}

fn expect_optional_usize(model_type: &str, params: &serde_yaml::Value, key: &str) -> Result<()> {
    if let Some(value) = model_param(params, key) {
        if value.as_u64().is_none() {
            candle_core::bail!(
                "model '{}' params.{} must be non-negative integer",
                model_type,
                key
            );
        }
    }
    Ok(())
}

fn expect_optional_bool(model_type: &str, params: &serde_yaml::Value, key: &str) -> Result<()> {
    if let Some(value) = model_param(params, key) {
        if value.as_bool().is_none() {
            candle_core::bail!("model '{}' params.{} must be bool", model_type, key);
        }
    }
    Ok(())
}

fn expect_optional_f64(model_type: &str, params: &serde_yaml::Value, key: &str) -> Result<()> {
    if let Some(value) = model_param(params, key) {
        if value.as_f64().is_none() {
            candle_core::bail!("model '{}' params.{} must be number", model_type, key);
        }
    }
    Ok(())
}

fn yaml_usize(v: &serde_yaml::Value, key: &str, default: usize) -> usize {
    v.get(key)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or(default)
}

fn yaml_usize_opt(v: &serde_yaml::Value, key: &str) -> Option<usize> {
    v.get(key).and_then(|v| v.as_u64()).map(|n| n as usize)
}

fn yaml_usize_seq(v: &serde_yaml::Value, key: &str) -> Vec<usize> {
    v.get(key)
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default()
}

fn yaml_string_seq(v: &serde_yaml::Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn yaml_f64(v: &serde_yaml::Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

fn parse_rankup_cross_token(
    value: &serde_yaml::Value,
) -> Result<rankup::tokenizer::CrossTokenConfig> {
    let map = value
        .as_mapping()
        .ok_or_else(|| candle_core::Error::Msg("rankup cross_token must be mapping".into()))?;
    let get = |key: &str| -> Result<String> {
        map.get(serde_yaml::Value::String(key.to_string()))
            .and_then(|value| value.as_str())
            .map(ToString::to_string)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("rankup cross_token.{key} must be string"))
            })
    };
    Ok(rankup::tokenizer::CrossTokenConfig {
        left: get("left")?,
        right: get("right")?,
    })
}

fn yaml_bool(v: &serde_yaml::Value, key: &str) -> bool {
    v.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn parse_task_config_entries(params: &serde_yaml::Value) -> Vec<TaskConfigEntry> {
    params
        .get("task_configs")
        .and_then(|v| v.as_sequence())
        .map(|seq| {
            seq.iter()
                .filter_map(|entry| {
                    let name = entry.get("name")?.as_str()?.to_string();
                    let tower_dims: Vec<usize> = yaml_usize_seq(entry, "tower_dims");
                    Some(TaskConfigEntry { name, tower_dims })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_multi_task_config(params: &serde_yaml::Value) -> Result<MultiTaskConfig> {
    match params.get("task_config") {
        Some(task_config) => serde_yaml::from_value::<MultiTaskConfig>(task_config.clone())
            .map_err(|e| {
                error!(error = %e, "parse unimixer task_config failed");
                candle_core::Error::Msg(format!("parse unimixer task_config: {}", e))
            }),
        None => Ok(MultiTaskConfig {
            towers: vec![],
            relations: vec![],
        }),
    }
}
