//! demo_inference：加载 Python 训练的权重，对 CSV 测试数据做推理，输出 logits。
//! 支持所有 5 种模型：LR, DeepFM, MMoE, ESMM, UniMixer。
use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::feats::config::FlowConfig;
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
use scale_rec::feats::ops::Fv;
use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;
use scale_rec::models::ModelConfig;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        eprintln!(
            "Usage: {} <feature_config.yaml> <model_config.yaml> <model.safetensors> <test.csv> <output.csv>",
            args[0]
        );
        std::process::exit(1);
    }
    let feature_config_path = &args[1];
    let model_config_path = &args[2];
    let safetensors_path = &args[3];
    let test_csv_path = &args[4];
    let output_csv_path = &args[5];

    // 1. 加载 feature config
    let yaml_str =
        std::fs::read_to_string(feature_config_path).context("Failed to read feature config")?;
    let flow_config = FlowConfig::from_yaml(&yaml_str).context("Invalid feature config YAML")?;
    let dag =
        FeatureDag::from_config(flow_config, false, None).map_err(|e| anyhow::anyhow!("{}", e))?;

    // 2. 获取 embeddable features
    let embed_features = dag.embeddable_features();
    let features: Vec<(String, usize, usize)> = embed_features
        .iter()
        .map(|(name, emb)| (name.to_string(), emb.vocab_size, emb.embed_dim))
        .collect();
    println!(
        "[Rust] {} embeddable features: {:?}",
        features.len(),
        features
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect::<Vec<_>>()
    );

    // 3. 加载 model config
    let model_yaml =
        std::fs::read_to_string(model_config_path).context("Failed to read model config")?;
    let model_config: ModelConfig =
        serde_yaml::from_str(&model_yaml).context("Invalid model config YAML")?;
    let model_type = model_config.model_type().to_string();
    println!("[Rust] model_type={}", model_type);

    // 4. 构建模型（注册 Var entries 到共享 VarMap） → 加载 safetensors
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // UniMixer 需要预构建 FeatureTokenizer（共享 VarMap，权重路径 "tokenizer.*"）
    let tokenizer: Option<FeatureTokenizer> = if model_type == "unimixer" {
        let token_dim = model_config
            .params
            .get("token_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or(64) as usize;
        let num_tokens = model_config
            .params
            .get("num_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        Some(FeatureTokenizer::new(
            vb.pp("tokenizer"),
            &features,
            token_dim,
            num_tokens,
        )?)
    } else {
        None
    };

    let model = model_config
        .build(vb, &features, tokenizer)
        .context("Failed to build model")?;
    varmap
        .load(safetensors_path)
        .context("Failed to load safetensors")?;
    println!("[Rust] loaded weights from {}", safetensors_path);

    // 6. 读取 CSV 并逐行 DAG 执行
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(test_csv_path)
        .context("Failed to open test CSV")?;
    let headers = reader
        .headers()
        .context("Failed to read CSV headers")?
        .clone();
    let col_index: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.to_string(), i))
        .collect();

    // Read source names from feature config to dynamically match CSV columns
    let source_names: Vec<String> = dag.source_defs().keys().cloned().collect();
    let embed_names: Vec<String> = features.iter().map(|(n, _, _)| n.clone()).collect();
    let mut all_indices: HashMap<String, Vec<u32>> = embed_names
        .iter()
        .map(|n| (n.clone(), Vec::new()))
        .collect();

    let mut row_count = 0usize;
    for result in reader.records() {
        let record = result.context("Failed to read CSV row")?;
        let mut raw_inputs: HashMap<String, FeatureValue> = HashMap::new();

        for field in &source_names {
            let col = col_index.get(field).context("Missing column")?;
            let val = record.get(*col).context("Missing field")?;
            let fv: FeatureValue = if val.parse::<f64>().is_ok() && val.contains('.') {
                Fv::Float(val.parse::<f32>().unwrap_or(0.0))
            } else if let Ok(i) = val.parse::<i64>() {
                Fv::Int(i as i32)
            } else {
                Fv::Str(val.to_string())
            };
            raw_inputs.insert(field.clone(), fv);
        }

        let pre_result = dag
            .execute(&raw_inputs)
            .map_err(|e| anyhow::anyhow!("DAG execute error: {}", e))?;

        for name in &embed_names {
            let val = pre_result
                .features
                .get(name)
                .with_context(|| format!("Feature '{}' not found in DAG output", name))?;
            let idx: i32 = match val.clone() {
                Fv::Int(i) => i,
                Fv::IntList(ref list) => list.first().copied().unwrap_or(0),
                _ => anyhow::bail!("Feature '{}' has unsupported type", name),
            };
            all_indices.get_mut(name).unwrap().push(idx as u32);
        }
        row_count += 1;
    }
    println!("[Rust] processed {} rows", row_count);

    // 7. 构建 batch tensor 并推理
    let tensor_inputs: HashMap<String, Tensor> = all_indices
        .iter()
        .map(|(name, indices)| {
            let tensor =
                Tensor::from_slice(indices.as_slice(), indices.len(), &Device::Cpu).unwrap();
            (name.clone(), tensor)
        })
        .collect();

    let outputs = model
        .forward(&tensor_inputs)
        .context("Model forward failed")?;

    // 8. 收集所有输出 key 并按名称排序
    let mut out_keys: Vec<&String> = outputs.keys().collect();
    out_keys.sort();

    let mut columns: Vec<String> = Vec::new();
    let mut data: Vec<Vec<f32>> = Vec::new();

    for key in &out_keys {
        let tensor = outputs.get(*key).context("Missing output")?;
        let vals: Vec<f32> = tensor
            .to_vec2::<f32>()
            .context("Failed to convert")?
            .iter()
            .map(|row| row[0])
            .collect();
        if columns.is_empty() {
            data.resize(vals.len(), Vec::new());
        }
        columns.push(format!("logit_{}", key));
        for (i, v) in vals.iter().enumerate() {
            data[i].push(*v);
        }
    }

    // 9. 写入输出 CSV
    let mut writer = csv::Writer::from_path(output_csv_path)?;
    writer.write_record(&columns)?;
    for row in &data {
        let row_strs: Vec<String> = row.iter().map(|v| format!("{:.8}", v)).collect();
        writer.write_record(&row_strs)?;
    }
    writer.flush()?;
    println!(
        "[Rust] wrote {} predictions (keys: {:?}) to {}",
        data.len(),
        out_keys,
        output_csv_path
    );
    Ok(())
}
