//! scale-rec 集成示例：特征预处理 + UniMixer 推理。
//! scale-rec integration example
mod feats;
mod layers;
mod models;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use feats::config::FlowConfig;
use feats::dag::{FeatureDag, FeatureValue};
use layers::towers::{Activation, MultiTaskConfig, TowerConfig};
use models::unimixer::model::UniMixerModel;
use models::unimixer::tokenizer::FeatureTokenizer;
use std::collections::HashMap;
use std::sync::Arc;

fn main() -> Result<()> {
    println!("=== scale-rec: FeatFlow + UniMixer ===");
    let yaml =
        std::fs::read_to_string("examples/feature_config.yaml").expect("Failed to read config");
    let flow_config = FlowConfig::from_yaml(&yaml).expect("Invalid YAML");
    println!("[Config] version={}", flow_config.version);

    let dag = FeatureDag::from_config(flow_config, true).map_err(|e| candle_core::Error::Msg(e))?;
    let embed_features = dag.embeddable_features();
    println!("\n[Embed] {} embeddable features:", embed_features.len());
    let tokenizer_features: Vec<(String, usize, usize)> = embed_features
        .iter()
        .map(|(name, emb)| {
            println!(
                "  {} -> vocab={}, dim={}",
                name, emb.vocab_size, emb.embed_dim
            );
            (name.to_string(), emb.vocab_size, emb.embed_dim)
        })
        .collect();

    println!("\n[Preprocess] Executing feature pipeline...");
    let mut raw_inputs: HashMap<String, FeatureValue> = HashMap::new();
    raw_inputs.insert("user_id".into(), Arc::new(42i32));
    raw_inputs.insert("user_age".into(), Arc::new(28.5f32));
    raw_inputs.insert("item_category".into(), Arc::new("electronics".to_string()));
    raw_inputs.insert(
        "user_tags".into(),
        Arc::new("sports#1|gaming#0.8".to_string()),
    );
    raw_inputs.insert("item_price".into(), Arc::new(5999.0f32));
    let pre_result = dag
        .execute(&raw_inputs)
        .map_err(|e| candle_core::Error::Msg(e))?;
    println!(
        "  Sources: {} | Computed: {}",
        pre_result.source_names.len(),
        pre_result.computed_names.len()
    );

    let batch_size = 1usize;
    let feature_tensors: HashMap<String, Tensor> = tokenizer_features
        .iter()
        .map(|(name, _, _)| {
            let val = pre_result.features.get(name).unwrap();
            let idx = *(**val).downcast_ref::<i32>().unwrap();
            let tensor = Tensor::from_slice(&[idx as u32], batch_size, &Device::Cpu).unwrap();
            (name.clone(), tensor)
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
    println!(
        "\n[Model] UniMixer: embed_dim={}, block_size={}",
        model.embed_dim, model.block_size
    );

    println!("\n[Forward] Running inference...");
    let outputs = model.forward_with_temperature(&feature_tensors, 0.5)?;
    for (name, logit) in &outputs {
        let val = logit.to_vec2::<f32>()?[0][0];
        println!("  {}: logit={:.4}", name, val);
    }
    Ok(())
}
