//! demo_inference：加载 Python 训练的权重，对 CSV 测试数据做推理，输出 logits。
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::feats::config::FlowConfig;
use scale_rec::feats::dag::{FeatureDag, FeatureValue};
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
        FeatureDag::from_config(flow_config, false).map_err(|e| anyhow::anyhow!("{}", e))?;

    // 2. 获取 embeddable features
    let embed_features = dag.embeddable_features();
    let features: Vec<(String, usize, usize)> = embed_features
        .iter()
        .map(|(name, emb)| (name.to_string(), emb.vocab_size, emb.embed_dim))
        .collect();
    println!(
        "[Rust] {} embeddable features: {:?}",
        features.len(),
        features.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>()
    );

    // 3. 加载 model config
    let model_yaml =
        std::fs::read_to_string(model_config_path).context("Failed to read model config")?;
    let model_config: ModelConfig =
        serde_yaml::from_str(&model_yaml).context("Invalid model config YAML")?;
    println!("[Rust] model_type={}", model_config.model_type());

    // 4. 构建模型（注册 Var entries）→ 加载 safetensors（填充权重）
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = model_config
        .build(vb, &features, None)
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
    let headers = reader.headers().context("Failed to read CSV headers")?.clone();
    let col_index: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.to_string(), i))
        .collect();

    let embed_names: Vec<String> = features.iter().map(|(n, _, _)| n.clone()).collect();
    let mut all_indices: HashMap<String, Vec<u32>> =
        embed_names.iter().map(|n| (n.clone(), Vec::new())).collect();

    let mut row_count = 0usize;
    for result in reader.records() {
        let record = result.context("Failed to read CSV row")?;
        let mut raw_inputs: HashMap<String, FeatureValue> = HashMap::new();

        for field in &["user_id", "user_age", "item_category", "user_tags", "item_price"] {
            let col = col_index.get(*field).context("Missing column")?;
            let val = record.get(*col).context("Missing field")?;
            let fv: FeatureValue = match *field {
                "user_id" => Arc::new(val.parse::<i32>().context("Invalid user_id")?),
                "user_age" => Arc::new(val.parse::<f32>().context("Invalid user_age")?),
                "item_category" | "user_tags" => Arc::new(val.to_string()),
                "item_price" => Arc::new(val.parse::<f32>().context("Invalid item_price")?),
                _ => continue,
            };
            raw_inputs.insert(field.to_string(), fv);
        }

        let pre_result = dag
            .execute(&raw_inputs)
            .map_err(|e| anyhow::anyhow!("DAG execute error: {}", e))?;

        for name in &embed_names {
            let val = pre_result
                .features
                .get(name)
                .with_context(|| format!("Feature '{}' not found in DAG output", name))?;
            let idx = *(**val)
                .downcast_ref::<i32>()
                .with_context(|| format!("Feature '{}' is not i32", name))?;
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
    let pred = outputs
        .get("pred")
        .context("Model output missing 'pred' key")?;
    let logits: Vec<f32> = pred
        .to_vec2::<f32>()
        .context("Failed to convert output to f32")?
        .iter()
        .map(|row| row[0])
        .collect();

    // 8. 写入输出 CSV
    let mut writer = csv::Writer::from_path(output_csv_path)?;
    writer.write_record(&["logit"])?;
    for &logit in &logits {
        writer.write_record(&[format!("{:.8}", logit)])?;
    }
    writer.flush()?;
    println!(
        "[Rust] wrote {} predictions to {}",
        logits.len(),
        output_csv_path
    );
    Ok(())
}
