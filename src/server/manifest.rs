//! 模型发布 manifest：权重、配置和版本元数据。
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// 权重绑定配置：格式、schema、路径前缀。
pub struct WeightBinding {
    /// 权重文件格式。
    #[serde(default = "default_weight_format")]
    pub format: String,
    /// 权重命名 schema。
    #[serde(default = "default_weight_schema")]
    pub schema: String,
    /// 模型权重根前缀。
    #[serde(default)]
    pub root_prefix: String,
    /// tokenizer 权重前缀。
    #[serde(default = "default_tokenizer_prefix")]
    pub tokenizer_prefix: String,
    /// UniMixer 主体权重前缀。
    #[serde(default = "default_unimixer_prefix")]
    pub unimixer_prefix: String,
    /// 是否严格要求 manifest 与权重 key 对齐。
    #[serde(default = "default_strict")]
    pub strict: bool,
    /// 是否允许 safetensors 中存在额外 tensor。
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
#[serde(deny_unknown_fields)]
/// 训练任务契约：任务名、标签、损失、权重、指标和输出语义。
pub struct TaskSpecManifest {
    /// 任务名称。
    pub name: String,
    /// 标签列名。
    pub label: String,
    /// 损失函数名称。
    #[serde(default = "default_task_loss")]
    pub loss: String,
    /// 任务损失权重。
    #[serde(default = "default_task_weight")]
    pub weight: f64,
    /// 可选 mask 列名。
    #[serde(default)]
    pub mask: Option<String>,
    /// 可选正样本权重。
    #[serde(default)]
    pub pos_weight: Option<f64>,
    /// 任务评估指标列表。
    #[serde(default)]
    pub metrics: Vec<String>,
    /// 模型输出语义类型。
    #[serde(default = "default_task_output_kind")]
    pub output_kind: String,
}

fn default_task_loss() -> String {
    "bce".into()
}

fn default_task_weight() -> f64 {
    1.0
}

fn default_task_output_kind() -> String {
    "binary_logit".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// 模型发布 Manifest：关联特征配置、模型配置和权重文件的元数据。
pub struct ModelManifest {
    /// manifest schema 版本。
    pub schema_version: u32,
    /// 逻辑模型 ID。
    pub model_id: String,
    /// 模型版本。
    pub model_version: String,
    /// 训练 run 版本。
    #[serde(default)]
    pub run_version: Option<String>,
    /// 发布版本。
    #[serde(default)]
    pub published_version: Option<String>,
    /// 模型结构类型。
    pub model_type: String,
    /// 产物对应的代码提交。
    pub code_commit: Option<String>,
    /// 权重文件路径，相对 manifest 所在目录或绝对路径。
    pub weights_file: String,
    /// 权重文件 sha256。
    pub weights_sha256: Option<String>,
    /// 权重命名绑定配置。
    #[serde(default)]
    pub weight_binding: WeightBinding,
    /// 特征配置文件路径。
    pub feature_config_file: String,
    /// 特征配置 sha256。
    pub feature_config_sha256: String,
    /// 模型配置文件路径。
    pub model_config_file: String,
    /// 模型配置 sha256。
    pub model_config_sha256: String,
    /// 参与训练和 serving 的基础任务名称。
    #[serde(default)]
    pub tasks: Vec<String>,
    /// 完整任务契约。
    #[serde(default)]
    pub task_specs: Vec<TaskSpecManifest>,
    /// 任务名到标签列的映射。
    #[serde(default)]
    pub label_col_map: HashMap<String, String>,
    /// 发布时记录的指标。
    #[serde(default)]
    pub metrics: HashMap<String, f64>,
    /// 最优 checkpoint 版本。
    #[serde(default)]
    pub best_version: Option<String>,
    /// 最优 checkpoint epoch。
    #[serde(default)]
    pub best_epoch: Option<u32>,
    /// 最优 checkpoint step。
    #[serde(default)]
    pub best_step: Option<u64>,
    /// 最优监控分数。
    #[serde(default)]
    pub best_score: Option<f64>,
    /// 最新 checkpoint 版本。
    #[serde(default)]
    pub latest_version: Option<String>,
    /// 最新 checkpoint epoch。
    #[serde(default)]
    pub latest_epoch: Option<u32>,
    /// 最新 checkpoint step。
    #[serde(default)]
    pub latest_step: Option<u64>,
    /// checkpoint 目录。
    #[serde(default)]
    pub checkpoint_dir: Option<String>,
    /// run manifest 文件路径。
    #[serde(default)]
    pub run_manifest_file: Option<String>,
    /// 发布权重文件路径。
    #[serde(default)]
    pub published_weights_file: Option<String>,
    /// best alias 权重文件路径。
    #[serde(default)]
    pub best_weights_file: Option<String>,
    /// latest alias 权重文件路径。
    #[serde(default)]
    pub latest_weights_file: Option<String>,
}

impl ModelManifest {
    /// 从 YAML 文件路径解析 Manifest。
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let yaml = std::fs::read_to_string(path)
            .map_err(|e| format!("read manifest {}: {}", path.display(), e))?;
        serde_yaml::from_str(&yaml).map_err(|e| format!("parse manifest {}: {}", path.display(), e))
    }

    /// 将相对路径解析为 manifest 所在目录下的绝对路径。
    pub fn resolve_from(&self, manifest_path: &Path, rel: &str) -> PathBuf {
        let normalized = rel.replace('\\', "/");
        let p = PathBuf::from(normalized);
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

/// 在模型目录中查找 manifest 文件。
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

/// 递归查找模型目录下的所有 manifest 文件。
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
            run_version: None,
            published_version: None,
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
            task_specs: vec![],
            label_col_map: HashMap::new(),
            metrics: HashMap::new(),
            best_version: None,
            best_epoch: None,
            best_step: None,
            best_score: None,
            latest_version: None,
            latest_epoch: None,
            latest_step: None,
            checkpoint_dir: None,
            run_manifest_file: None,
            published_weights_file: None,
            best_weights_file: None,
            latest_weights_file: None,
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
