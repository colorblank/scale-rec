//! 模型发布 manifest：权重、配置和版本元数据。
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub model_version: String,
    pub model_type: String,
    pub code_commit: Option<String>,
    pub weights_file: String,
    pub weights_sha256: Option<String>,
    pub feature_config_file: String,
    pub feature_config_sha256: String,
    pub model_config_file: String,
    pub model_config_sha256: String,
    #[serde(default)]
    pub tasks: Vec<String>,
    #[serde(default)]
    pub label_col_map: HashMap<String, String>,
    #[serde(default)]
    pub metrics: HashMap<String, f64>,
}

impl ModelManifest {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let yaml = std::fs::read_to_string(path)
            .map_err(|e| format!("read manifest {}: {}", path.display(), e))?;
        serde_yaml::from_str(&yaml).map_err(|e| format!("parse manifest {}: {}", path.display(), e))
    }

    pub fn resolve_from(&self, manifest_path: &Path, rel: &str) -> PathBuf {
        let p = PathBuf::from(rel);
        if p.is_absolute() {
            p
        } else {
            manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(p)
        }
    }
}

pub fn find_manifest(model_dir: &Path, model_name: &str) -> Option<PathBuf> {
    [
        model_dir.join(format!("{}.manifest.yaml", model_name)),
        model_dir.join(format!("{}_manifest.yaml", model_name)),
        model_dir.join(model_name).join("model_manifest.yaml"),
        model_dir.join(format!("{}.yaml", model_name)),
    ]
    .into_iter()
    .find(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_paths_against_manifest_dir() {
        let manifest = ModelManifest {
            schema_version: 1,
            model_id: "m".into(),
            model_version: "v".into(),
            model_type: "lr".into(),
            code_commit: None,
            weights_file: "m.safetensors".into(),
            weights_sha256: None,
            feature_config_file: "feature.yaml".into(),
            feature_config_sha256: "abc".into(),
            model_config_file: "model.yaml".into(),
            model_config_sha256: "def".into(),
            tasks: vec![],
            label_col_map: HashMap::new(),
            metrics: HashMap::new(),
        };

        let path = manifest.resolve_from(Path::new("/tmp/model/m.manifest.yaml"), "m.safetensors");

        assert_eq!(path, PathBuf::from("/tmp/model/m.safetensors"));
    }
}
