//! 模型注册表：多模型管理 + 热更新。
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::Serialize;
use tracing::info;

use super::engine::InferenceEngine;
use crate::feats::config::FlowConfig;
use crate::feats::dag::FeatureDag;
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use crate::models::ModelConfig;

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub loaded_at: String,
}

/// 线程安全的多模型注册表。
pub struct ModelRegistry {
    engines: RwLock<HashMap<String, Arc<InferenceEngine>>>,
    feature_config_path: PathBuf,
    model_dir: PathBuf,
    embed_features_cache: RwLock<Vec<(String, usize, usize)>>,
}

impl ModelRegistry {
    pub fn new(feature_config_path: &Path, model_dir: &Path) -> Result<Self, String> {
        let yaml = std::fs::read_to_string(feature_config_path)
            .map_err(|e| format!("read feature config: {}", e))?;
        let flow_config = FlowConfig::from_yaml(&yaml).map_err(|e| format!("parse: {}", e))?;
        let dag = FeatureDag::from_config(flow_config, false, None)
            .map_err(|e| format!("dag: {}", e))?;
        let embed_features = dag.embeddable_features();
        let features: Vec<(String, usize, usize)> = embed_features
            .iter()
            .map(|(n, e)| (n.to_string(), e.vocab_size, e.embed_dim))
            .collect();
        Ok(Self {
            engines: RwLock::new(HashMap::new()),
            feature_config_path: feature_config_path.to_path_buf(),
            model_dir: model_dir.to_path_buf(),
            embed_features_cache: RwLock::new(features),
        })
    }

    /// 加载或重载指定模型。
    pub fn load_model(&self, model_name: &str) -> Result<ModelInfo, String> {
        let safetensors_path = self.model_dir.join(format!("{}.safetensors", model_name));
        if !safetensors_path.exists() {
            return Err(format!("model file not found: {}", safetensors_path.display()));
        }

        let model_config_path = self.find_model_config(model_name)?;
        let model_yaml = std::fs::read_to_string(&model_config_path)
            .map_err(|e| format!("read model config: {}", e))?;
        let model_config: ModelConfig =
            serde_yaml::from_str(&model_yaml).map_err(|e| format!("parse: {}", e))?;
        let model_type = model_config.model_type().to_string();

        let yaml = std::fs::read_to_string(&self.feature_config_path)
            .map_err(|e| format!("read feature config: {}", e))?;
        let flow_config = FlowConfig::from_yaml(&yaml).map_err(|e| format!("parse: {}", e))?;
        let dag = FeatureDag::from_config(flow_config, false, None)
            .map_err(|e| format!("dag: {}", e))?;
        let embed_names: Vec<String> = dag
            .embeddable_features()
            .iter()
            .map(|(n, _)| n.to_string())
            .collect();

        let cached_features = self.embed_features_cache.read().unwrap();

        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let tokenizer: Option<FeatureTokenizer> = if model_type == "unimixer" {
            let td = model_config
                .params
                .get("token_dim")
                .and_then(|v| v.as_u64())
                .unwrap_or(64) as usize;
            let nt = model_config
                .params
                .get("num_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as usize;
            Some(FeatureTokenizer::new(vb.pp("tokenizer"), &cached_features, td, nt)
                .map_err(|e| format!("tokenizer: {}", e))?)
        } else {
            None
        };

        let model = model_config
            .build(vb, &cached_features, tokenizer)
            .map_err(|e| format!("build: {}", e))?;
        varmap
            .load(safetensors_path.to_str().unwrap_or(""))
            .map_err(|e| format!("load weights: {}", e))?;

        let engine = Arc::new(InferenceEngine {
            dag,
            model,
            embed_names,
        });

        let mut engines = self.engines.write().unwrap();
        engines.insert(model_name.to_string(), engine);
        drop(engines);

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        info!("[registry] loaded model '{}'", model_name);

        Ok(ModelInfo {
            name: model_name.to_string(),
            loaded_at: ts.to_string(),
        })
    }

    fn find_model_config(&self, model_name: &str) -> Result<PathBuf, String> {
        // If model_name starts with "model_", use it as-is for config naming
        let (base, suffix) = if let Some(stripped) = model_name.strip_prefix("model_") {
            (model_name, stripped)  // model_lr → base=model_lr, suffix=lr
        } else {
            (model_name.as_ref(), model_name) // lr → base=lr, suffix=lr
        };
        let candidates = vec![
            self.model_dir.parent().map(|p| p.join(format!("{}_demo.yaml", base))),
            Some(PathBuf::from(format!("python/config/model_{}.yaml", suffix))),
            Some(PathBuf::from(format!("python/demo/model_{}_demo.yaml", suffix))),
            Some(PathBuf::from(format!("python/demo/{}_demo.yaml", base))),
        ];
        for c in candidates.into_iter().flatten() {
            if c.exists() {
                return Ok(c);
            }
        }
        Err(format!(
            "model config not found for '{}' (tried demo/ and config/)",
            model_name
        ))
    }

    pub fn get(&self, name: &str) -> Option<Arc<InferenceEngine>> {
        self.engines.read().unwrap().get(name).cloned()
    }

    pub fn list(&self) -> Vec<String> {
        self.engines.read().unwrap().keys().cloned().collect()
    }
}
