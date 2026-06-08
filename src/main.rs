//! scale-rec 集成示例：特征预处理 + UniMixer 推理。
//! scale-rec integration example
mod feats;
mod layers;
mod models;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use feats::config::FlowConfig;
use feats::dag::{FeatureDag, FeatureValue};
use feats::ops::Fv;
use layers::embedding::FeatureSpec;
use layers::towers::{Activation, MultiTaskConfig, TowerConfig};
use models::unimixer::model::UniMixerModel;
use models::unimixer::tokenizer::FeatureTokenizer;
use std::collections::HashMap;
use tracing::info;
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    info!("scale-rec: FeatFlow + UniMixer");
    let yaml = std::fs::read_to_string("examples/feature_config_discover.yaml")
        .expect("Failed to read config");
    let flow_config = FlowConfig::from_yaml(&yaml).expect("Invalid YAML");
    info!(version = %flow_config.version, "config loaded");

    let dag =
        FeatureDag::from_config(flow_config, true, None).map_err(|e| candle_core::Error::Msg(e))?;
    let embed_features = dag.embeddable_features();
    info!(count = embed_features.len(), "embeddable features");
    let tokenizer_features: Vec<FeatureSpec> = embed_features
        .iter()
        .map(|(name, emb)| {
            info!(name = %name, vocab = emb.vocab_size, dim = emb.embed_dim, "feature spec");
            FeatureSpec {
                name: name.to_string(),
                vocab_size: emb.vocab_size,
                embed_dim: emb.embed_dim,
                pooling: emb.pooling,
                seq_len: emb.seq_len.or_else(|| {
                    dag.feature_schemas
                        .get(*name)
                        .and_then(|schema| schema.dtype.list_len())
                }),
                truncation: emb.truncation,
            }
        })
        .collect();

    info!("executing feature pipeline");
    let mut raw_inputs: HashMap<String, FeatureValue> = HashMap::new();
    raw_inputs.insert("user_id".into(), Fv::Int(42));
    raw_inputs.insert("user_age".into(), Fv::Float(28.5));
    raw_inputs.insert("item_category".into(), Fv::Str("electronics".into()));
    raw_inputs.insert("user_tags".into(), Fv::Str("sports#1|gaming#0.8".into()));
    raw_inputs.insert("item_price".into(), Fv::Float(5999.0));
    let pre_result = dag
        .execute(&raw_inputs)
        .map_err(|e| candle_core::Error::Msg(e))?;
    info!(
        sources = pre_result.source_names.len(),
        computed = pre_result.computed_names.len(),
        "feature pipeline executed"
    );

    let batch_size = 1usize;
    let feature_tensors: HashMap<String, Tensor> = tokenizer_features
        .iter()
        .map(|spec| {
            let val = pre_result.features.get(&spec.name).unwrap();
            let idx: i32 = match val.clone() {
                Fv::Int(i) => i,
                Fv::IntList(ref list) => list.first().copied().unwrap_or(0),
                _ => panic!("Feature '{}' has unsupported type", spec.name),
            };
            let tensor = Tensor::from_slice(&[idx as u32], batch_size, &Device::Cpu).unwrap();
            (spec.name.clone(), tensor)
        })
        .collect();

    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let token_dim = 64;
    let num_tokens = 8;
    let tokenizer = FeatureTokenizer::new(
        vb.pp("tokenizer"),
        &tokenizer_features,
        token_dim,
        num_tokens,
    )?;
    let task_config = MultiTaskConfig {
        towers: vec![
            TowerConfig {
                name: "ctr".into(),
                hidden_dims: vec![128],
                output_dim: 1,
                activation: Activation::Relu,
            },
            TowerConfig {
                name: "cvr".into(),
                hidden_dims: vec![128],
                output_dim: 1,
                activation: Activation::Relu,
            },
        ],
        relations: vec![],
    };
    let model = UniMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        2,
        None,
        false,
        1.0,
        4,
        16,
        &task_config,
        false,
        vb.pp("unimixer"),
    )?;
    info!(
        embed_dim = model.embed_dim,
        block_size = model.block_size,
        "UniMixer model built"
    );

    info!("running inference");
    let outputs = model.forward_with_temperature(&feature_tensors, 0.5)?;
    for (name, logit) in &outputs {
        let val = logit.to_vec2::<f32>()?[0][0];
        info!(task = %name, logit = val, "prediction");
    }
    Ok(())
}
