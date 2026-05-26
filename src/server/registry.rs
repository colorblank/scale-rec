//! 模型注册表：多模型管理 + 热更新。
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{safetensors::MmapedSafetensors, DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::Serialize;
use tracing::{info, warn};

use super::engine::InferenceEngine;
use super::manifest::{find_manifest, ModelManifest};
use crate::feats::config::FlowConfig;
use crate::feats::dag::FeatureDag;
use crate::layers::embedding::FeatureSpec;
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use crate::models::ModelConfig;

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub loaded_at: String,
    pub model_version: Option<String>,
    pub manifest_path: Option<String>,
}

/// 线程安全的多模型注册表。
pub struct ModelRegistry {
    engines: RwLock<HashMap<String, Arc<InferenceEngine>>>,
    feature_config_path: PathBuf,
    model_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(feature_config_path: &Path, model_dir: &Path) -> Result<Self, String> {
        if !feature_config_path.exists() {
            return Err(format!(
                "feature config path not found: {}",
                feature_config_path.display()
            ));
        }
        Ok(Self {
            engines: RwLock::new(HashMap::new()),
            feature_config_path: feature_config_path.to_path_buf(),
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// 加载或重载指定模型。
    pub fn load_model(&self, model_name: &str) -> Result<ModelInfo, String> {
        let manifest_path = find_manifest(&self.model_dir, model_name);
        let manifest = manifest_path
            .as_deref()
            .map(ModelManifest::from_path)
            .transpose()?;

        let safetensors_path = match (&manifest, &manifest_path) {
            (Some(m), Some(p)) => m.resolve_from(p, &m.weights_file),
            _ => self.model_dir.join(format!("{}.safetensors", model_name)),
        };
        if !safetensors_path.exists() {
            return Err(format!(
                "model file not found: {}",
                safetensors_path.display()
            ));
        }

        let model_config_path = match (&manifest, &manifest_path) {
            (Some(m), Some(p)) => m.resolve_from(p, &m.model_config_file),
            _ => self.find_model_config(model_name)?,
        };
        if let Some(m) = &manifest {
            self.validate_manifest_files(m, &model_config_path, &safetensors_path)?;
        }
        let model_yaml = std::fs::read_to_string(&model_config_path)
            .map_err(|e| format!("read model config: {}", e))?;
        let model_config: ModelConfig =
            serde_yaml::from_str(&model_yaml).map_err(|e| format!("parse: {}", e))?;
        let model_type = model_config.model_type().to_string();
        if let Some(m) = &manifest {
            if m.model_type != model_type {
                return Err(format!(
                    "manifest model_type '{}' does not match model config '{}'",
                    m.model_type, model_type
                ));
            }
        }

        let yaml = std::fs::read_to_string(&self.feature_config_path)
            .map_err(|e| format!("read feature config: {}", e))?;
        let flow_config = FlowConfig::from_yaml(&yaml).map_err(|e| format!("parse: {}", e))?;
        let dag =
            FeatureDag::from_config(flow_config, false, None).map_err(|e| format!("dag: {}", e))?;
        let embed_features: Vec<FeatureSpec> = dag
            .embeddable_features()
            .iter()
            .map(|(n, e)| FeatureSpec {
                name: n.to_string(),
                vocab_size: e.vocab_size,
                embed_dim: e.embed_dim,
                pooling: e.pooling,
                seq_len: e.seq_len,
            })
            .collect();

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
            Some(
                FeatureTokenizer::new(vb.pp("tokenizer"), &embed_features, td, nt)
                    .map_err(|e| format!("tokenizer: {}", e))?,
            )
        } else {
            None
        };

        let model = model_config
            .build(vb, &embed_features, tokenizer)
            .map_err(|e| format!("build: {}", e))?;
        validate_safetensors_keys(&varmap, &safetensors_path)?;
        varmap
            .load(&safetensors_path)
            .map_err(|e| format!("load weights: {}", e))?;

        let engine = Arc::new(InferenceEngine::new(dag, model, embed_features));

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
            model_version: manifest.as_ref().map(|m| m.model_version.clone()),
            manifest_path: manifest_path.map(|p| p.display().to_string()),
        })
    }

    fn validate_manifest_files(
        &self,
        manifest: &ModelManifest,
        model_config_path: &Path,
        safetensors_path: &Path,
    ) -> Result<(), String> {
        let feature_sha = sha256_file(&self.feature_config_path)?;
        if feature_sha != manifest.feature_config_sha256 {
            return Err(format!(
                "feature config sha256 mismatch: runtime={} manifest={}",
                feature_sha, manifest.feature_config_sha256
            ));
        }
        let model_sha = sha256_file(model_config_path)?;
        if model_sha != manifest.model_config_sha256 {
            return Err(format!(
                "model config sha256 mismatch: runtime={} manifest={}",
                model_sha, manifest.model_config_sha256
            ));
        }
        if let Some(expected_weights_sha) = &manifest.weights_sha256 {
            let weights_sha = sha256_file(safetensors_path)?;
            if &weights_sha != expected_weights_sha {
                return Err(format!(
                    "weights sha256 mismatch: runtime={} manifest={}",
                    weights_sha, expected_weights_sha
                ));
            }
        }
        if manifest.schema_version != 1 {
            return Err(format!(
                "unsupported manifest schema_version {}",
                manifest.schema_version
            ));
        }
        Ok(())
    }

    fn find_model_config(&self, model_name: &str) -> Result<PathBuf, String> {
        // Look next to model_dir first, then alongside feature_config.
        // Demo configs may live under python/configs/demo/{legacy,discover}.
        let demo_parent = self.model_dir.parent();
        let feature_parent = self.feature_config_path.parent();
        let parent_candidates: Vec<Option<&Path>> = vec![demo_parent, feature_parent];
        let model_key = model_name.strip_prefix("model_").unwrap_or(model_name);
        let (config_key, prefers_discover) = match model_key.strip_prefix("discover_") {
            Some(key) => (key, true),
            None => (model_key, false),
        };
        let config_names = vec![
            format!("{}_demo.yaml", model_name),
            format!("model_{}_demo.yaml", model_key),
            format!("model_{}.yaml", model_key),
            format!("model_{}.yaml", config_key),
        ];
        for parent in parent_candidates.into_iter().flatten() {
            let dirs = match parent.file_name().and_then(|name| name.to_str()) {
                Some("legacy") | Some("discover") => {
                    let config_dir = parent.parent();
                    let sibling = config_dir.map(|dir| {
                        dir.join(if parent.file_name().unwrap() == "legacy" {
                            "discover"
                        } else {
                            "legacy"
                        })
                    });
                    match (
                        prefers_discover,
                        parent.file_name().and_then(|name| name.to_str()),
                    ) {
                        (true, Some("legacy")) | (false, Some("discover")) => {
                            sibling.into_iter().chain([parent.to_path_buf()]).collect()
                        }
                        _ => [parent.to_path_buf()].into_iter().chain(sibling).collect(),
                    }
                }
                _ => vec![
                    parent.to_path_buf(),
                    parent.join("configs").join(if prefers_discover {
                        "discover"
                    } else {
                        "legacy"
                    }),
                    parent.join("configs").join(if prefers_discover {
                        "legacy"
                    } else {
                        "discover"
                    }),
                ],
            };
            for dir in dirs {
                for name in &config_names {
                    let p = dir.join(name);
                    if p.exists() {
                        return Ok(p);
                    }
                }
            }
        }
        Err(format!("model config not found for '{}'", model_name))
    }

    pub fn get(&self, name: &str) -> Option<Arc<InferenceEngine>> {
        self.engines.read().unwrap().get(name).cloned()
    }

    pub fn list(&self) -> Vec<String> {
        self.engines.read().unwrap().keys().cloned().collect()
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let digest = sha256_bytes(&bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        write!(&mut out, "{:02x}", b).unwrap();
    }
    Ok(out)
}

fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_len = (input.len() as u64) * 8;
    let mut msg = input.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let j = i * 4;
            *word = u32::from_be_bytes([chunk[j], chunk[j + 1], chunk[j + 2], chunk[j + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn validate_safetensors_keys(varmap: &VarMap, path: &Path) -> Result<(), String> {
    let expected: HashMap<String, Vec<usize>> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().dims().to_vec()))
        .collect();
    let safetensors = unsafe { MmapedSafetensors::new(path) }
        .map_err(|e| format!("read safetensors header {}: {}", path.display(), e))?;
    let tensors = safetensors.tensors();
    let actual: HashMap<String, Vec<usize>> = tensors
        .iter()
        .map(|(name, view)| (name.clone(), view.shape().to_vec()))
        .collect();

    let actual_names: HashSet<&String> = actual.keys().collect();
    let expected_names: HashSet<&String> = expected.keys().collect();
    let missing: Vec<&String> = expected_names.difference(&actual_names).copied().collect();
    if !missing.is_empty() {
        return Err(format!("missing safetensors keys: {:?}", missing));
    }

    for (name, expected_shape) in &expected {
        let actual_shape = actual.get(name).unwrap();
        if actual_shape != expected_shape {
            return Err(format!(
                "safetensors shape mismatch for '{}': expected {:?}, got {:?}",
                name, expected_shape, actual_shape
            ));
        }
    }

    let extra: Vec<&String> = actual_names.difference(&expected_names).copied().collect();
    if !extra.is_empty() {
        warn!("extra safetensors keys ignored: {:?}", extra);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::fs;

    fn hex(bytes: [u8; 32]) -> String {
        let mut out = String::with_capacity(64);
        for b in bytes {
            write!(&mut out, "{:02x}", b).unwrap();
        }
        out
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            hex(sha256_bytes(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(sha256_bytes(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn empty_registry(feature_config_path: PathBuf, model_dir: PathBuf) -> ModelRegistry {
        ModelRegistry {
            engines: RwLock::new(HashMap::new()),
            feature_config_path,
            model_dir,
        }
    }

    #[test]
    fn finds_model_config_in_demo_config_subdirs() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-registry-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let python_dir = root.join("python");
        let artifact_dir = python_dir.join("artifacts").join("demo");
        let config_dir = python_dir.join("configs").join("demo");
        let legacy_dir = config_dir.join("legacy");
        let discover_dir = config_dir.join("discover");
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::create_dir_all(&legacy_dir).unwrap();
        fs::create_dir_all(&discover_dir).unwrap();

        let feature_config = legacy_dir.join("feature_config.yaml");
        let legacy_lr = legacy_dir.join("model_lr.yaml");
        let legacy_esmm = legacy_dir.join("model_esmm.yaml");
        let discover_esmm = discover_dir.join("model_esmm.yaml");
        fs::write(&feature_config, "").unwrap();
        fs::write(&legacy_lr, "type: lr\n").unwrap();
        fs::write(&legacy_esmm, "type: esmm\n").unwrap();
        fs::write(&discover_esmm, "type: esmm\n").unwrap();

        let registry = empty_registry(feature_config, artifact_dir);
        assert_eq!(registry.find_model_config("model_lr").unwrap(), legacy_lr);
        assert_eq!(
            registry.find_model_config("model_discover_esmm").unwrap(),
            discover_esmm
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validate_manifest_files_checks_optional_weights_sha256() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-manifest-sha-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let feature_config = root.join("feature.yaml");
        let model_config = root.join("model.yaml");
        let weights = root.join("model.safetensors");
        fs::write(&feature_config, "feature").unwrap();
        fs::write(&model_config, "model").unwrap();
        fs::write(&weights, "weights").unwrap();

        let registry = empty_registry(feature_config.clone(), root.clone());
        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "m".into(),
            model_version: "v".into(),
            model_type: "lr".into(),
            code_commit: None,
            weights_file: "model.safetensors".into(),
            weights_sha256: Some(hex(sha256_bytes(b"different"))),
            feature_config_file: "feature.yaml".into(),
            feature_config_sha256: hex(sha256_bytes(b"feature")),
            model_config_file: "model.yaml".into(),
            model_config_sha256: hex(sha256_bytes(b"model")),
            tasks: vec![],
            label_col_map: HashMap::new(),
            metrics: HashMap::new(),
        };

        let err = registry
            .validate_manifest_files(&manifest, &model_config, &weights)
            .unwrap_err();
        assert!(err.contains("weights sha256 mismatch"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_feature_hot_reload_consistency() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-reload-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let feature_config = root.join("feature.yaml");
        let model_config = root.join("model_lr_demo.yaml");
        let weights = root.join("model_lr.safetensors");

        // 1. Initial configuration
        let initial_feature_yaml = "version: '1.0.0'\nsources:\n  - name: user_id\n    dtype: int\n    default_val: '0'\noperators:\n  - name: user_id_hash\n    op_type: FeatureHash\n    inputs: [user_id]\n    outputs: [user_id_idx]\n    params: {vocab_size: 100, num_hashes: 1}\n    embed: {vocab_size: 100, embed_dim: 8}\n";
        fs::write(&feature_config, initial_feature_yaml).unwrap();
        fs::write(&model_config, "type: lr\n").unwrap();

        let device = Device::Cpu;
        let mut tensors1 = HashMap::new();
        tensors1.insert(
            "global_bias".to_string(),
            Tensor::zeros((1,), DType::F32, &device).unwrap(),
        );
        tensors1.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((100, 8), DType::F32, &device).unwrap(),
        );
        tensors1.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), DType::F32, &device).unwrap(),
        );
        tensors1.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), DType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors1, &weights).unwrap();

        let registry = ModelRegistry::new(&feature_config, &root).unwrap();
        registry.load_model("model_lr").unwrap();

        // Check registry engine initialized with vocab_size = 100
        {
            let engine = registry.get("model_lr").unwrap();
            assert_eq!(engine.embed_features[0].vocab_size, 100);
        }

        // 2. Hot reload with modified feature configuration and new weights
        let updated_feature_yaml = "version: '1.0.0'\nsources:\n  - name: user_id\n    dtype: int\n    default_val: '0'\noperators:\n  - name: user_id_hash\n    op_type: FeatureHash\n    inputs: [user_id]\n    outputs: [user_id_idx]\n    params: {vocab_size: 120, num_hashes: 1}\n    embed: {vocab_size: 120, embed_dim: 8}\n";
        fs::write(&feature_config, updated_feature_yaml).unwrap();

        let mut tensors2 = HashMap::new();
        tensors2.insert(
            "global_bias".to_string(),
            Tensor::zeros((1,), DType::F32, &device).unwrap(),
        );
        tensors2.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((120, 8), DType::F32, &device).unwrap(),
        );
        tensors2.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), DType::F32, &device).unwrap(),
        );
        tensors2.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), DType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors2, &weights).unwrap();

        // Load again
        registry.load_model("model_lr").unwrap();

        // Check registry engine is re-initialized with vocab_size = 120
        {
            let engine = registry.get("model_lr").unwrap();
            assert_eq!(engine.embed_features[0].vocab_size, 120);
        }

        fs::remove_dir_all(root).unwrap();
    }
}
