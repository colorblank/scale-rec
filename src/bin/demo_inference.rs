//! demo_inference：加载 Python 训练的权重，对 CSV 测试数据做推理，输出 logits。
//! 支持所有注册模型：LR, DeepFM, MMoE, ESMM, GDCN+ESMM, UniMixer。
use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use scale_rec::feats::builder::DagBuilder;
use scale_rec::feats::config::FlowConfig;
use scale_rec::feats::executor::DagExecutor;
use scale_rec::feats::feature_info::FeatureInfo;
use scale_rec::layers::embedding::FeatureSpec;
use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;
use scale_rec::models::ModelConfig;
use scale_rec::server::engine::{FeatureRow, InferenceEngine};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        error!(
            "usage: {} <feature_config.yaml> <model_config.yaml> <model.safetensors> <test.csv> <output.csv>",
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
    let artifact = DagBuilder::build(flow_config).map_err(|e| anyhow::anyhow!("{}", e))?;
    let feat_info = FeatureInfo::new(
        artifact.sources.clone(),
        artifact.node_defs.clone(),
        artifact.execution_order.clone(),
    );

    // 2. 获取 embeddable features
    let embed_features = feat_info.embeddable_features();
    let features: Vec<FeatureSpec> = embed_features
        .iter()
        .map(|(name, emb)| FeatureSpec {
            name: name.to_string(),
            vocab_size: emb.vocab_size,
            embed_dim: emb.embed_dim,
            pooling: emb.pooling,
            seq_len: emb.seq_len.or_else(|| {
                artifact
                    .feature_schemas
                    .get(*name)
                    .and_then(|schema| schema.dtype.list_len())
            }),
            truncation: emb.truncation,
        })
        .collect();
    info!(
        count = features.len(),
        names = ?features.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
        "embeddable features"
    );

    // 3. 加载 model config
    let model_yaml =
        std::fs::read_to_string(model_config_path).context("Failed to read model config")?;
    let model_config: ModelConfig =
        serde_yaml::from_str(&model_yaml).context("Invalid model config YAML")?;
    let model_type = model_config.model_type().to_string();
    info!(model_type = %model_type, "model config loaded");

    // 4. 构建模型（注册 Var entries 到共享 VarMap） → 加载 safetensors
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Token-based models need a shared FeatureTokenizer at path "tokenizer.*".
    let tokenizer: Option<FeatureTokenizer> = if model_type == "unimixer"
        || model_type == "token_mixer_large"
        || model_type == "rankmixer"
        || model_type == "full_mix"
    {
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
    model.warmup().context("Failed to warm up model caches")?;
    info!(path = %safetensors_path, "loaded weights");

    let op_kind = feat_info.op_source_kind();
    let user_ops: std::collections::HashSet<String> = op_kind
        .iter()
        .filter(|(_, &k)| !k.has_item())
        .map(|(n, _)| n.clone())
        .collect();
    let executor = DagExecutor::new(
        artifact.plan,
        artifact.sources,
        artifact.execution_order,
        artifact.data_sources,
    );
    let user_op_indices: std::collections::HashSet<usize> = executor
        .plan()
        .steps
        .iter()
        .filter(|s| user_ops.contains(&executor.execution_order()[s.op_idx]))
        .map(|s| s.op_idx)
        .collect();
    let broadcast_precompute_skip_indices: std::collections::HashSet<usize> = executor
        .plan()
        .steps
        .iter()
        .filter(|s| !user_ops.contains(&executor.execution_order()[s.op_idx]))
        .map(|s| s.op_idx)
        .collect();
    let engine = InferenceEngine::new(
        executor,
        model,
        features,
        device,
        user_op_indices,
        broadcast_precompute_skip_indices,
    );

    // 5. 读取 CSV，并复用服务端 InferenceEngine 的 source-aware 解析和 tensor 构造。
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

    let source_names: Vec<String> = engine.executor.source_defs().keys().cloned().collect();
    let mut rows: Vec<FeatureRow> = Vec::new();

    for result in reader.records() {
        let record = result.context("Failed to read CSV row")?;
        let mut row = HashMap::new();
        for field in &source_names {
            let col = col_index.get(field).context("Missing column")?;
            let val = record.get(*col).context("Missing field")?;
            row.insert(field.clone(), serde_json::Value::String(val.to_string()));
        }
        rows.push(row);
    }
    info!(rows = rows.len(), "processed input rows");

    let (predictions, _metrics) = engine
        .predict(&rows)
        .map_err(|e| anyhow::anyhow!("Inference failed: {}", e))?;

    // 6. 收集所有输出 key 并按名称排序
    let mut out_keys: Vec<&String> = predictions
        .first()
        .map(|row| row.keys().collect())
        .unwrap_or_default();
    out_keys.sort();

    let mut columns: Vec<String> = Vec::new();
    for key in &out_keys {
        columns.push(format!("logit_{}", key));
    }

    // 7. 写入输出 CSV
    let mut writer = csv::Writer::from_path(output_csv_path)?;
    writer.write_record(&columns)?;
    for row in &predictions {
        let row_strs: Vec<String> = out_keys
            .iter()
            .map(|key| format!("{:.8}", row.get(*key).copied().unwrap_or_default()))
            .collect();
        writer.write_record(&row_strs)?;
    }
    writer.flush()?;
    info!(
        predictions = predictions.len(),
        keys = ?out_keys,
        output = %output_csv_path,
        "wrote predictions"
    );
    Ok(())
}
