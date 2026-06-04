//! 模型发布 manifest：权重、配置和版本元数据。
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightBinding {
    #[serde(default = "default_weight_format")]
    pub format: String,
    #[serde(default = "default_weight_schema")]
    pub schema: String,
    #[serde(default)]
    pub root_prefix: String,
    #[serde(default = "default_tokenizer_prefix")]
    pub tokenizer_prefix: String,
    #[serde(default = "default_unimixer_prefix")]
    pub unimixer_prefix: String,
    #[serde(default = "default_strict")]
    pub strict: bool,
    #[serde(default = "default_allow_extra_tensors")]
    pub allow_extra_tensors: bool,
}

impl Default for WeightBinding {
    fn default() -> Self {
        Self {
            format: default_weight_format(),
            schema: default_weight_schema(),
            root_prefix: String::new(),
            tokenizer_prefix: default_tokenizer_prefix(),
            unimixer_prefix: default_unimixer_prefix(),
            strict: default_strict(),
            allow_extra_tensors: default_allow_extra_tensors(),
        }
    }
}

fn default_weight_format() -> String {
    "safetensors".into()
}

fn default_weight_schema() -> String {
    "candle-varbuilder-v1".into()
}

fn default_tokenizer_prefix() -> String {
    "tokenizer".into()
}

fn default_unimixer_prefix() -> String {
    "unimixer".into()
}

fn default_strict() -> bool {
    true
}

fn default_allow_extra_tensors() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema_version: u32,
    pub model_id: String,
    pub model_version: String,
    pub model_type: String,
    pub code_commit: Option<String>,
    pub weights_file: String,
    pub weights_sha256: Option<String>,
    #[serde(default)]
    pub weight_binding: WeightBinding,
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

pub fn find_manifests(model_dir: &Path) -> Vec<PathBuf> {
    let mut manifests = Vec::new();
    collect_manifests(model_dir, 0, &mut manifests);
    manifests.sort();
    manifests.dedup();
    manifests
}

fn collect_manifests(dir: &Path, depth: usize, manifests: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_manifests(&path, depth + 1, manifests);
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if file_name == "run.manifest.yaml" {
            continue;
        }
        if file_name == "model_manifest.yaml"
            || file_name.ends_with(".manifest.yaml")
            || file_name.ends_with("_manifest.yaml")
        {
            manifests.push(path);
        }
    }
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
            weight_binding: WeightBinding::default(),
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

    #[test]
    fn parses_default_weight_binding_for_older_manifests() {
        let yaml = r#"
schema_version: 1
model_id: m
model_version: v
model_type: lr
code_commit:
weights_file: m.safetensors
weights_sha256:
feature_config_file: feature.yaml
feature_config_sha256: abc
model_config_file: model.yaml
model_config_sha256: def
"#;

        let manifest: ModelManifest = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(manifest.weight_binding, WeightBinding::default());
    }

    #[test]
    fn parses_custom_weight_binding() {
        let yaml = r#"
schema_version: 1
model_id: m
model_version: v
model_type: unimixer
code_commit:
weights_file: m.safetensors
weights_sha256:
weight_binding:
  format: safetensors
  schema: candle-varbuilder-v1
  root_prefix: serving
  tokenizer_prefix: feature_tokenizer
  unimixer_prefix: scorer
  strict: false
  allow_extra_tensors: true
feature_config_file: feature.yaml
feature_config_sha256: abc
model_config_file: model.yaml
model_config_sha256: def
"#;

        let manifest: ModelManifest = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(manifest.weight_binding.root_prefix, "serving");
        assert_eq!(
            manifest.weight_binding.tokenizer_prefix,
            "feature_tokenizer"
        );
        assert_eq!(manifest.weight_binding.unimixer_prefix, "scorer");
        assert!(!manifest.weight_binding.strict);
        assert!(manifest.weight_binding.allow_extra_tensors);
    }

    #[test]
    fn finds_manifests_in_root_and_version_dirs() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-find-manifests-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let version_dir = root.join("ranker").join("20260604");
        std::fs::create_dir_all(&version_dir).unwrap();
        let root_manifest = root.join("ranker.manifest.yaml");
        let run_manifest = version_dir.join("run.manifest.yaml");
        let version_manifest = version_dir.join("model_manifest.yaml");
        let config_yaml = root.join("model.yaml");
        std::fs::write(&root_manifest, "").unwrap();
        std::fs::write(&run_manifest, "").unwrap();
        std::fs::write(&version_manifest, "").unwrap();
        std::fs::write(&config_yaml, "").unwrap();

        let manifests = find_manifests(&root);

        assert!(manifests.contains(&root_manifest));
        assert!(manifests.contains(&version_manifest));
        assert!(!manifests.contains(&run_manifest));
        assert!(!manifests.contains(&config_yaml));

        std::fs::remove_dir_all(root).unwrap();
    }
}
