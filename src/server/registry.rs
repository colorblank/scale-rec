//! 模型注册表：多模型管理 + 热更新。
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{safetensors::MmapedSafetensors, DType as CandleDType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use super::engine::InferenceEngine;
use super::manifest::{find_manifest, ModelManifest, TaskSpecManifest, WeightBinding};
use crate::feats::builder::DagBuilder;
use crate::feats::config::{DType, DataSourceDef, FlowConfig, SourceKind};
use crate::feats::executor::DagExecutor;
use crate::feats::feature_info::FeatureInfo;
use crate::layers::embedding::FeatureSpec;
use crate::models::unimixer::tokenizer::FeatureTokenizer;
use crate::models::{ModelBuildOptions, ModelConfig};

/// 模型摘要信息。
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    /// 逻辑模型名称。
    pub name: String,
    /// 加载时间戳。
    pub loaded_at: String,
    /// 已加载模型版本。
    pub model_version: Option<String>,
    /// manifest 文件路径。
    pub manifest_path: Option<String>,
}

/// 模型版本详细信息。
#[derive(Debug, Clone, Serialize)]
pub struct ModelVersionInfo {
    /// 版本号。
    pub version: String,
    /// 加载时间戳。
    pub loaded_at: String,
    /// 模型结构类型。
    pub model_type: String,
    /// manifest 文件路径。
    pub manifest_path: Option<String>,
    /// 是否为默认版本。
    pub is_default: bool,
    /// schema hash，当前使用 feature config sha256。
    pub schema_hash: Option<String>,
    /// 特征配置 sha256。
    pub feature_config_sha256: Option<String>,
    /// 模型配置 sha256。
    pub model_config_sha256: Option<String>,
    /// 基础任务名称列表。
    pub tasks: Vec<String>,
    /// 完整任务契约。
    pub task_specs: Vec<TaskSpecManifest>,
    /// 任务名到标签列映射。
    pub label_col_map: HashMap<String, String>,
    /// 发布指标。
    pub metrics: HashMap<String, f64>,
    /// 权重绑定配置。
    pub weight_binding: WeightBinding,
}

/// 模型别名映射。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAliasInfo {
    /// 别名名称。
    pub alias: String,
    /// 别名指向的版本。
    pub version: String,
}

/// 加权流量路由版本。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedVersion {
    /// 参与加权路由的版本。
    pub version: String,
    /// 路由权重。
    pub weight: u32,
}

/// 流量路由策略：固定版本或加权分发。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RoutingPolicy {
    /// 固定路由到一个版本。
    Fixed {
        /// 固定版本号。
        version: String,
    },
    /// 按权重在多个版本间路由。
    Weighted {
        /// 用于稳定哈希的请求字段。
        #[serde(default)]
        key_field: Option<String>,
        /// 稳定哈希盐值。
        #[serde(default)]
        salt: Option<String>,
        /// 加权版本列表。
        versions: Vec<WeightedVersion>,
    },
}

/// 模型 serving 完整信息。
#[derive(Debug, Clone, Serialize)]
pub struct ModelServingInfo {
    /// 逻辑模型名称。
    pub name: String,
    /// 默认版本加载时间。
    pub loaded_at: Option<String>,
    /// 默认版本 schema hash。
    pub schema_hash: Option<String>,
    /// 默认版本基础任务列表。
    pub tasks: Vec<String>,
    /// 默认版本完整任务契约。
    pub task_specs: Vec<TaskSpecManifest>,
    /// 默认版本任务名到标签列映射。
    pub label_col_map: HashMap<String, String>,
    /// 默认版本发布指标。
    pub metrics: HashMap<String, f64>,
    /// 默认版本权重绑定配置。
    pub weight_binding: Option<WeightBinding>,
    /// 默认版本号。
    pub default_version: Option<String>,
    /// 模型别名列表。
    pub aliases: Vec<ModelAliasInfo>,
    /// 当前路由策略。
    pub routing: Option<RoutingPolicy>,
    /// 已加载版本列表。
    pub versions: Vec<ModelVersionInfo>,
}

/// 特征输入描述。
#[derive(Debug, Clone, Serialize)]
pub struct FeatureInputInfo {
    /// 输入字段名称。
    pub name: String,
    /// 输入字段业务来源。
    pub source: Option<SourceKind>,
    /// 输入字段的数据源名称。
    pub data_source: Option<String>,
    /// 输入字段类型。
    pub dtype: DType,
    /// 输入字段缺省值。
    pub default_val: String,
}

/// 特征契约：模型声明的输入特征和 data sources。
#[derive(Debug, Clone, Serialize)]
pub struct FeatureContract {
    /// 逻辑模型名称。
    pub model: String,
    /// 模型版本。
    pub version: String,
    /// 模型使用的数据源列表。
    pub data_sources: Vec<DataSourceDef>,
    /// 推理请求需要提供的输入字段。
    pub required_inputs: Vec<FeatureInputInfo>,
}

#[derive(Clone)]
struct LoadedModelVersion {
    engine: Arc<InferenceEngine>,
    info: ModelVersionInfo,
}

#[derive(Default)]
struct ModelEntry {
    default_version: Option<String>,
    versions: HashMap<String, LoadedModelVersion>,
    aliases: HashMap<String, String>,
    routing: Option<RoutingPolicy>,
}

/// 解析后的模型引用：包含引擎及版本信息。
#[derive(Clone)]
pub struct ResolvedModel {
    /// 已解析出的推理引擎。
    pub engine: Arc<InferenceEngine>,
    /// 逻辑模型名称。
    pub model_name: String,
    /// 实际选中的版本。
    pub version: String,
}

/// 线程安全的多模型注册表。
pub struct ModelRegistry {
    models: RwLock<HashMap<String, ModelEntry>>,
    feature_config_path: Option<PathBuf>,
    model_dir: PathBuf,
}

impl ModelRegistry {
    /// 创建注册表（指定特征配置路径和模型目录）。
    pub fn new(feature_config_path: &Path, model_dir: &Path) -> Result<Self, String> {
        if !feature_config_path.exists() {
            return Err(format!(
                "feature config path not found: {}",
                feature_config_path.display()
            ));
        }
        Ok(Self {
            models: RwLock::new(HashMap::new()),
            feature_config_path: Some(feature_config_path.to_path_buf()),
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// 从模型目录创建注册表（需通过 manifest 加载）。
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, String> {
        if !model_dir.exists() {
            return Err(format!("model dir not found: {}", model_dir.display()));
        }
        Ok(Self {
            models: RwLock::new(HashMap::new()),
            feature_config_path: None,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// 重载模型。
    pub fn load_model(&self, model_name: &str) -> Result<ModelInfo, String> {
        let manifest_path = find_manifest(&self.model_dir, model_name);
        self.load_resolved_model(model_name, manifest_path, None, None)
    }

    /// 从 manifest 文件加载模型。
    pub fn load_manifest(&self, manifest_path: &Path) -> Result<ModelInfo, String> {
        let manifest = ModelManifest::from_path(manifest_path)?;
        let model_name = manifest.model_id.clone();
        self.load_resolved_model(&model_name, Some(manifest_path.to_path_buf()), None, None)
    }

    /// 直接加载 safetensors 文件及其关联配置。
    pub fn load_safetensors(&self, safetensors_path: &Path) -> Result<ModelInfo, String> {
        if !safetensors_path.exists() {
            return Err(format!(
                "model file not found: {}",
                safetensors_path.display()
            ));
        }
        if !safetensors_path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "safetensors")
            .unwrap_or(false)
        {
            return Err(format!(
                "not a safetensors file: {}",
                safetensors_path.display()
            ));
        }
        let model_name = safetensors_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("invalid model file name: {}", safetensors_path.display()))?;
        self.load_resolved_model(
            model_name,
            None,
            Some(safetensors_path.to_path_buf()),
            safetensors_path.parent().map(Path::to_path_buf),
        )
    }

    fn load_resolved_model(
        &self,
        model_name: &str,
        manifest_path: Option<PathBuf>,
        safetensors_path_override: Option<PathBuf>,
        config_search_dir: Option<PathBuf>,
    ) -> Result<ModelInfo, String> {
        let manifest = manifest_path
            .as_deref()
            .map(ModelManifest::from_path)
            .transpose()?;

        let safetensors_path = match (&manifest, &manifest_path) {
            (Some(m), Some(p)) => m.resolve_from(p, &m.weights_file),
            _ => safetensors_path_override
                .unwrap_or_else(|| self.model_dir.join(format!("{}.safetensors", model_name))),
        };
        if !safetensors_path.exists() {
            return Err(format!(
                "model file not found: {}",
                safetensors_path.display()
            ));
        }

        let model_config_path = match (&manifest, &manifest_path) {
            (Some(m), Some(p)) => m.resolve_from(p, &m.model_config_file),
            _ => self.find_model_config_in(model_name, config_search_dir.as_deref())?,
        };
        let feature_config_path = match (&manifest, &manifest_path) {
            (Some(m), Some(p)) => m.resolve_from(p, &m.feature_config_file),
            _ => self.feature_config_path.clone().ok_or_else(|| {
                "feature config is required when loading without manifest".to_string()
            })?,
        };
        if let Some(m) = &manifest {
            self.validate_manifest_files(
                m,
                &feature_config_path,
                &model_config_path,
                &safetensors_path,
            )?;
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
        let weight_binding = manifest
            .as_ref()
            .map(|m| m.weight_binding.clone())
            .unwrap_or_default();
        validate_weight_binding(&weight_binding)?;

        let yaml = std::fs::read_to_string(&feature_config_path)
            .map_err(|e| format!("read feature config: {}", e))?;
        let flow_config = FlowConfig::from_yaml(&yaml).map_err(|e| format!("parse: {}", e))?;
        let artifact = DagBuilder::build(flow_config).map_err(|e| format!("dag: {}", e))?;
        let feat_info = FeatureInfo::new(
            artifact.sources.clone(),
            artifact.node_defs.clone(),
            artifact.execution_order.clone(),
        );
        let embed_features: Vec<FeatureSpec> = feat_info
            .embeddable_features()
            .iter()
            .map(|(n, e)| FeatureSpec {
                name: n.to_string(),
                vocab_size: e.vocab_size,
                embed_dim: e.embed_dim,
                pooling: e.pooling,
                seq_len: e.seq_len.or_else(|| {
                    artifact
                        .feature_schemas
                        .get(*n)
                        .and_then(|schema| schema.dtype.list_len())
                }),
                truncation: e.truncation,
            })
            .collect();

        let device = {
            #[cfg(feature = "macos-metal")]
            {
                let dev = Device::new_metal(0).unwrap_or(Device::Cpu);
                info!("[registry] selected device: {:?}", dev);
                dev
            }
            #[cfg(not(feature = "macos-metal"))]
            {
                info!("[registry] selected device: Cpu");
                Device::Cpu
            }
        };
        let mut varmap = VarMap::new();
        let base_vb = VarBuilder::from_varmap(&varmap, CandleDType::F32, &device);
        let vb = scoped_vb(base_vb, &weight_binding.root_prefix);

        let tokenizer: Option<FeatureTokenizer> = if model_requires_feature_tokenizer(&model_type) {
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
                FeatureTokenizer::new(
                    scoped_vb(vb.clone(), &weight_binding.tokenizer_prefix),
                    &embed_features,
                    td,
                    nt,
                )
                .map_err(|e| format!("tokenizer: {}", e))?,
            )
        } else {
            None
        };

        let build_options = ModelBuildOptions {
            unimixer_prefix: weight_binding.unimixer_prefix.clone(),
        };
        let model = model_config
            .build_with_options(vb, &embed_features, tokenizer, &build_options)
            .map_err(|e| format!("build: {}", e))?;
        validate_safetensors_keys(&varmap, &safetensors_path, &weight_binding)?;
        varmap
            .load(&safetensors_path)
            .map_err(|e| format!("load weights: {}", e))?;
        model
            .warmup()
            .map_err(|e| format!("warm up model caches: {}", e))?;

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("system clock before unix epoch: {}", e))?
            .as_secs();
        let loaded_at = ts.to_string();
        let version = manifest
            .as_ref()
            .map(|m| m.model_version.clone())
            .unwrap_or_else(|| "default".to_string());
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
        let engine = Arc::new(InferenceEngine::new(
            executor,
            model,
            embed_features,
            device,
            user_op_indices,
            broadcast_precompute_skip_indices,
        ));
        let info = ModelVersionInfo {
            version: version.clone(),
            loaded_at: loaded_at.clone(),
            model_type: model_type.clone(),
            manifest_path: manifest_path.as_ref().map(|p| p.display().to_string()),
            is_default: false,
            schema_hash: manifest.as_ref().map(|m| m.feature_config_sha256.clone()),
            feature_config_sha256: manifest.as_ref().map(|m| m.feature_config_sha256.clone()),
            model_config_sha256: manifest.as_ref().map(|m| m.model_config_sha256.clone()),
            tasks: manifest
                .as_ref()
                .map(|m| m.tasks.clone())
                .unwrap_or_default(),
            task_specs: manifest
                .as_ref()
                .map(|m| m.task_specs.clone())
                .unwrap_or_default(),
            label_col_map: manifest
                .as_ref()
                .map(|m| m.label_col_map.clone())
                .unwrap_or_default(),
            metrics: manifest
                .as_ref()
                .map(|m| m.metrics.clone())
                .unwrap_or_default(),
            weight_binding: weight_binding.clone(),
        };

        let mut models = self
            .models
            .write()
            .map_err(|e| format!("model registry lock poisoned: {}", e))?;
        let entry = models.entry(model_name.to_string()).or_default();
        if entry
            .default_version
            .as_ref()
            .map(|current| version > *current)
            .unwrap_or(true)
        {
            entry.default_version = Some(version.clone());
        }
        entry.versions.insert(
            version.clone(),
            LoadedModelVersion {
                engine,
                info: info.clone(),
            },
        );
        drop(models);

        info!(
            "[registry] loaded model '{}' version '{}'",
            model_name, version
        );

        Ok(ModelInfo {
            name: model_name.to_string(),
            loaded_at,
            model_version: Some(version),
            manifest_path: manifest_path.map(|p| p.display().to_string()),
        })
    }

    fn validate_manifest_files(
        &self,
        manifest: &ModelManifest,
        feature_config_path: &Path,
        model_config_path: &Path,
        safetensors_path: &Path,
    ) -> Result<(), String> {
        let feature_sha = sha256_file(feature_config_path)?;
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

    fn find_model_config_in(
        &self,
        model_name: &str,
        extra_parent: Option<&Path>,
    ) -> Result<PathBuf, String> {
        // Look in model_dir first, then alongside feature_config.
        // Canonical demo/example configs now live under examples/.
        let feature_parent = self
            .feature_config_path
            .as_ref()
            .and_then(|path| path.parent());
        let mut parent_candidates: Vec<&Path> = Vec::new();
        if let Some(parent) = extra_parent {
            parent_candidates.push(parent);
        }
        parent_candidates.push(self.model_dir.as_path());
        parent_candidates.push(feature_parent.unwrap_or_else(|| Path::new(".")));
        let model_key = model_name.strip_prefix("model_").unwrap_or(model_name);
        let config_key = model_key.strip_prefix("demo_").unwrap_or(model_key);
        let demo_key = model_key.strip_suffix("_demo").unwrap_or(model_key);
        let config_names = vec![
            format!("model_{}.yaml", model_key),
            format!("model_{}.yaml", config_key),
            format!("model_{}.yaml", demo_key),
            format!("{}.yaml", model_key),
            format!("{}.yaml", config_key),
            format!("{}.yaml", demo_key),
            format!("model_{}_demo.yaml", model_key),
            format!("model_{}_demo.yaml", config_key),
        ];
        for parent in parent_candidates {
            for name in &config_names {
                let p = parent.join(name);
                if p.exists() {
                    return Ok(p);
                }
            }
        }
        Err(format!("model config not found for '{}'", model_name))
    }

    /// 获取模型引擎。
    pub fn get(&self, name: &str) -> Option<Arc<InferenceEngine>> {
        self.resolve(name, None, None)
            .map(|resolved| resolved.engine)
    }

    /// 根据名称和版本解析模型。
    pub fn resolve(
        &self,
        name: &str,
        version: Option<&str>,
        fallback_version: Option<&str>,
    ) -> Option<ResolvedModel> {
        self.resolve_request(name, version, None, fallback_version, None)
    }

    /// 完整解析流程：版本/别名/路由策略/回退版本。
    pub fn resolve_request(
        &self,
        name: &str,
        version: Option<&str>,
        alias: Option<&str>,
        fallback_version: Option<&str>,
        routing_key: Option<&str>,
    ) -> Option<ResolvedModel> {
        let models = self.models.read().ok()?;
        let entry = models.get(name)?;
        let selected_version =
            self.select_version(entry, version, alias, fallback_version, routing_key)?;
        let loaded = entry.versions.get(&selected_version)?;
        Some(ResolvedModel {
            engine: loaded.engine.clone(),
            model_name: name.to_string(),
            version: selected_version,
        })
    }

    /// 返回已加载的模型名称列表。
    pub fn list(&self) -> Vec<String> {
        let Ok(models) = self.models.read() else {
            warn!("model registry lock poisoned while listing models");
            return vec![];
        };
        let mut names: Vec<String> = models.keys().cloned().collect();
        names.sort();
        names
    }

    /// 返回所有模型的 serving 信息。
    pub fn list_info(&self) -> Vec<ModelServingInfo> {
        let Ok(models) = self.models.read() else {
            warn!("model registry lock poisoned while listing model info");
            return vec![];
        };
        let mut result: Vec<ModelServingInfo> = models
            .iter()
            .map(|(name, entry)| self.entry_info(name, entry))
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// 返回指定模型的 serving 信息。
    pub fn model_info(&self, name: &str) -> Option<ModelServingInfo> {
        let models = self.models.read().ok()?;
        models.get(name).map(|entry| self.entry_info(name, entry))
    }

    /// 返回指定模型的别名列表。
    pub fn aliases(&self, name: &str) -> Option<Vec<ModelAliasInfo>> {
        let models = self.models.read().ok()?;
        models.get(name).map(|entry| {
            let mut aliases: Vec<ModelAliasInfo> = entry
                .aliases
                .iter()
                .map(|(alias, version)| ModelAliasInfo {
                    alias: alias.clone(),
                    version: version.clone(),
                })
                .collect();
            aliases.sort_by(|a, b| a.alias.cmp(&b.alias));
            aliases
        })
    }

    /// 返回指定模型的路由策略。
    pub fn routing_policy(&self, name: &str) -> Option<Option<RoutingPolicy>> {
        let models = self.models.read().ok()?;
        models.get(name).map(|entry| entry.routing.clone())
    }

    /// 设置模型版本别名。
    pub fn set_alias(&self, name: &str, alias: &str, version: &str) -> Result<(), String> {
        let mut models = self
            .models
            .write()
            .map_err(|e| format!("model registry lock poisoned: {}", e))?;
        let entry = models
            .get_mut(name)
            .ok_or_else(|| format!("model '{}' not found", name))?;
        if !entry.versions.contains_key(version) {
            return Err(format!(
                "version '{}' not found for model '{}'",
                version, name
            ));
        }
        entry.aliases.insert(alias.to_string(), version.to_string());
        Ok(())
    }

    /// 删除模型版本别名。
    pub fn delete_alias(&self, name: &str, alias: &str) -> Result<(), String> {
        let mut models = self
            .models
            .write()
            .map_err(|e| format!("model registry lock poisoned: {}", e))?;
        let entry = models
            .get_mut(name)
            .ok_or_else(|| format!("model '{}' not found", name))?;
        entry.aliases.remove(alias);
        Ok(())
    }

    /// 设置模型路由策略。
    pub fn set_routing_policy(
        &self,
        name: &str,
        routing: Option<RoutingPolicy>,
    ) -> Result<(), String> {
        let mut models = self
            .models
            .write()
            .map_err(|e| format!("model registry lock poisoned: {}", e))?;
        let entry = models
            .get_mut(name)
            .ok_or_else(|| format!("model '{}' not found", name))?;
        if let Some(policy) = &routing {
            self.validate_routing_policy(entry, policy, name)?;
        }
        entry.routing = routing;
        Ok(())
    }

    /// 返回指定模型的特征契约。
    pub fn feature_contract(&self, name: &str, version: Option<&str>) -> Option<FeatureContract> {
        let resolved = self.resolve(name, version, None)?;
        let mut required_inputs: Vec<FeatureInputInfo> = resolved
            .engine
            .executor
            .source_defs()
            .values()
            .map(|source| FeatureInputInfo {
                name: source.name.clone(),
                source: source.source.clone(),
                data_source: source.data_source.clone(),
                dtype: source.dtype.clone(),
                default_val: source.default_val.clone(),
            })
            .collect();
        required_inputs.sort_by(|a, b| a.name.cmp(&b.name));
        Some(FeatureContract {
            model: resolved.model_name,
            version: resolved.version,
            data_sources: resolved.engine.executor.data_sources().to_vec(),
            required_inputs,
        })
    }

    fn entry_info(&self, name: &str, entry: &ModelEntry) -> ModelServingInfo {
        let mut versions: Vec<ModelVersionInfo> = entry
            .versions
            .iter()
            .map(|(version, loaded)| {
                let mut info = loaded.info.clone();
                info.is_default = entry
                    .default_version
                    .as_ref()
                    .map(|default| default == version)
                    .unwrap_or(false);
                info
            })
            .collect();
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        let mut aliases: Vec<ModelAliasInfo> = entry
            .aliases
            .iter()
            .map(|(alias, version)| ModelAliasInfo {
                alias: alias.clone(),
                version: version.clone(),
            })
            .collect();
        aliases.sort_by(|a, b| a.alias.cmp(&b.alias));
        let default_info = entry
            .default_version
            .as_ref()
            .and_then(|version| entry.versions.get(version))
            .map(|loaded| &loaded.info);
        ModelServingInfo {
            name: name.to_string(),
            loaded_at: default_info.map(|info| info.loaded_at.clone()),
            schema_hash: default_info.and_then(|info| info.schema_hash.clone()),
            tasks: default_info
                .map(|info| info.tasks.clone())
                .unwrap_or_default(),
            task_specs: default_info
                .map(|info| info.task_specs.clone())
                .unwrap_or_default(),
            label_col_map: default_info
                .map(|info| info.label_col_map.clone())
                .unwrap_or_default(),
            metrics: default_info
                .map(|info| info.metrics.clone())
                .unwrap_or_default(),
            weight_binding: default_info.map(|info| info.weight_binding.clone()),
            default_version: entry.default_version.clone(),
            aliases,
            routing: entry.routing.clone(),
            versions,
        }
    }

    fn select_version(
        &self,
        entry: &ModelEntry,
        version: Option<&str>,
        alias: Option<&str>,
        fallback_version: Option<&str>,
        routing_key: Option<&str>,
    ) -> Option<String> {
        let selected = if let Some(version) = version {
            Some(version.to_string())
        } else if let Some(alias) = alias {
            entry.aliases.get(alias).cloned()
        } else if let Some(routing) = &entry.routing {
            self.select_routing_version(entry, routing, routing_key)
        } else {
            entry.default_version.clone()
        }?;

        if entry.versions.contains_key(&selected) {
            return Some(selected);
        }

        fallback_version
            .filter(|v| entry.versions.contains_key(*v))
            .map(str::to_string)
    }

    fn select_routing_version(
        &self,
        entry: &ModelEntry,
        routing: &RoutingPolicy,
        routing_key: Option<&str>,
    ) -> Option<String> {
        match routing {
            RoutingPolicy::Fixed { version } => Some(version.clone()),
            RoutingPolicy::Weighted {
                key_field,
                salt,
                versions,
            } => {
                let selected = self.select_weighted_version(
                    entry,
                    versions,
                    routing_key,
                    key_field.as_deref(),
                    salt.as_deref(),
                )?;
                Some(selected)
            }
        }
    }

    fn select_weighted_version(
        &self,
        entry: &ModelEntry,
        versions: &[WeightedVersion],
        routing_key: Option<&str>,
        key_field: Option<&str>,
        salt: Option<&str>,
    ) -> Option<String> {
        let total_weight: u64 = versions
            .iter()
            .filter(|candidate| {
                candidate.weight > 0 && entry.versions.contains_key(&candidate.version)
            })
            .map(|candidate| candidate.weight as u64)
            .sum();
        if total_weight == 0 {
            return None;
        }
        let key = routing_key?;
        let mut bytes = Vec::new();
        if let Some(field) = key_field {
            bytes.extend_from_slice(field.as_bytes());
            bytes.push(0);
        }
        if let Some(salt) = salt {
            bytes.extend_from_slice(salt.as_bytes());
            bytes.push(0);
        }
        bytes.extend_from_slice(key.as_bytes());
        let hash = sha256_bytes(&bytes);
        let mut ticket = u64::from_be_bytes(hash[0..8].try_into().ok()?);
        ticket %= total_weight;
        for candidate in versions {
            if candidate.weight == 0 {
                continue;
            }
            if !entry.versions.contains_key(&candidate.version) {
                continue;
            }
            let weight = candidate.weight as u64;
            if ticket < weight {
                return Some(candidate.version.clone());
            }
            ticket -= weight;
        }
        None
    }

    fn validate_routing_policy(
        &self,
        entry: &ModelEntry,
        policy: &RoutingPolicy,
        model_name: &str,
    ) -> Result<(), String> {
        let versions = match policy {
            RoutingPolicy::Fixed { version } => vec![version.as_str()],
            RoutingPolicy::Weighted { versions, .. } => {
                if versions.is_empty() {
                    return Err(format!(
                        "routing policy for model '{}' must include at least one version",
                        model_name
                    ));
                }
                versions.iter().map(|v| v.version.as_str()).collect()
            }
        };
        for version in versions {
            if !entry.versions.contains_key(version) {
                return Err(format!(
                    "routing policy references unknown version '{}' for model '{}'",
                    version, model_name
                ));
            }
        }
        Ok(())
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let digest = sha256_bytes(&bytes);
    Ok(hex_encode(&digest))
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

fn scoped_vb<'a>(vb: VarBuilder<'a>, prefix: &str) -> VarBuilder<'a> {
    if prefix.is_empty() {
        vb
    } else {
        vb.pp(prefix)
    }
}

fn model_requires_feature_tokenizer(model_type: &str) -> bool {
    matches!(
        model_type,
        "unimixer" | "token_mixer_large" | "rankmixer" | "full_mix"
    )
}

fn validate_weight_binding(binding: &WeightBinding) -> Result<(), String> {
    if binding.format != "safetensors" {
        return Err(format!("unsupported weight format '{}'", binding.format));
    }
    if binding.schema != "candle-varbuilder-v1" {
        return Err(format!("unsupported weight schema '{}'", binding.schema));
    }
    Ok(())
}

fn validate_safetensors_keys(
    varmap: &VarMap,
    path: &Path,
    binding: &WeightBinding,
) -> Result<(), String> {
    let expected: HashMap<String, Vec<usize>> = varmap
        .data()
        .lock()
        .map_err(|e| format!("varmap lock poisoned: {}", e))?
        .iter()
        .map(|(name, var)| (name.clone(), var.as_tensor().dims().to_vec()))
        .collect();
    // SAFETY: The path is an existing safetensors file selected from a manifest or explicit
    // model path. The mapping is read-only and is only used to inspect tensor metadata here.
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
        if binding.strict {
            return Err(format!("missing safetensors keys: {:?}", missing));
        }
        warn!(
            "missing safetensors keys left at initializer values: {:?}",
            missing
        );
    }

    for (name, expected_shape) in &expected {
        let Some(actual_shape) = actual.get(name) else {
            continue;
        };
        if actual_shape != expected_shape {
            return Err(format!(
                "safetensors shape mismatch for '{}': expected {:?}, got {:?}",
                name, expected_shape, actual_shape
            ));
        }
    }

    let extra: Vec<&String> = actual_names.difference(&expected_names).copied().collect();
    if !extra.is_empty() {
        if !binding.allow_extra_tensors {
            return Err(format!("extra safetensors keys: {:?}", extra));
        }
        warn!("extra safetensors keys ignored: {:?}", extra);
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::fs;

    fn hex(bytes: [u8; 32]) -> String {
        hex_encode(&bytes)
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
            models: RwLock::new(HashMap::new()),
            feature_config_path: Some(feature_config_path),
            model_dir,
        }
    }

    #[test]
    fn identifies_models_requiring_feature_tokenizer() {
        for model_type in ["unimixer", "token_mixer_large", "rankmixer", "full_mix"] {
            assert!(model_requires_feature_tokenizer(model_type));
        }
        for model_type in ["lr", "deepfm", "mmoe", "esmm", "gdcn_esmm"] {
            assert!(!model_requires_feature_tokenizer(model_type));
        }
    }

    #[test]
    fn finds_model_config_in_examples_dir() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-registry-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let artifact_dir = root.join("artifacts").join("demo");
        let config_dir = root.join("examples");
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::create_dir_all(&config_dir).unwrap();

        let feature_config = config_dir.join("feature_config_demo.yaml");
        let demo_gdcn_esmm = config_dir.join("model_demo_gdcn_esmm.yaml");
        let demo_unimixer = config_dir.join("model_demo_unimixer.yaml");
        fs::write(&feature_config, "").unwrap();
        fs::write(&demo_gdcn_esmm, "type: gdcn_esmm\n").unwrap();
        fs::write(&demo_unimixer, "type: unimixer\n").unwrap();

        let registry = empty_registry(feature_config, artifact_dir);
        assert_eq!(
            registry
                .find_model_config_in("model_demo_gdcn_esmm", None)
                .unwrap(),
            demo_gdcn_esmm
        );
        assert_eq!(
            registry
                .find_model_config_in("model_demo_unimixer", None)
                .unwrap(),
            demo_unimixer
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
            run_version: None,
            published_version: None,
            model_type: "lr".into(),
            code_commit: None,
            weights_file: "model.safetensors".into(),
            weights_sha256: Some(hex(sha256_bytes(b"different"))),
            weight_binding: WeightBinding::default(),
            feature_config_file: "feature.yaml".into(),
            feature_config_sha256: hex(sha256_bytes(b"feature")),
            model_config_file: "model.yaml".into(),
            model_config_sha256: hex(sha256_bytes(b"model")),
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
            embedding_bucket_report_file: None,
        };

        let err = registry
            .validate_manifest_files(&manifest, &feature_config, &model_config, &weights)
            .unwrap_err();
        assert!(err.contains("weights sha256 mismatch"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_model_from_manifest_feature_config_without_global_feature_config() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-manifest-only-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let feature_config = root.join("feature.yaml");
        let model_config = root.join("model.yaml");
        let weights = root.join("ranker.safetensors");
        let manifest_path = root.join("ranker.manifest.yaml");

        let feature_yaml = "version: '1.0.0'\ndata_sources:\n  - name: user_profile_hbase\n    kind: hbase\nsources:\n  - name: user_id\n    source: User\n    data_source: user_profile_hbase\n    dtype: int\n    default_val: '0'\noperators:\n  - name: user_id_hash\n    op_type: FeatureHash\n    inputs: [user_id]\n    outputs: [user_id_idx]\n    params: {vocab_size: 100, num_hashes: 1}\n    embed: {vocab_size: 100, embed_dim: 8}\n";
        let model_yaml = "type: lr\n";
        fs::write(&feature_config, feature_yaml).unwrap();
        fs::write(&model_config, model_yaml).unwrap();

        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((100, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &weights).unwrap();

        let manifest = format!(
            "schema_version: 1\nmodel_id: ranker\nmodel_version: v1\nmodel_type: lr\ncode_commit:\nweights_file: ranker.safetensors\nweights_sha256: {}\nfeature_config_file: feature.yaml\nfeature_config_sha256: {}\nmodel_config_file: model.yaml\nmodel_config_sha256: {}\n",
            sha256_file(&weights).unwrap(),
            sha256_file(&feature_config).unwrap(),
            sha256_file(&model_config).unwrap(),
        );
        fs::write(&manifest_path, manifest).unwrap();

        let registry = ModelRegistry::from_model_dir(&root).unwrap();
        let info = registry.load_model("ranker").unwrap();

        assert_eq!(info.name, "ranker");
        assert!(registry.get("ranker").is_some());
        let contract = registry.feature_contract("ranker", Some("v1")).unwrap();
        assert_eq!(contract.model, "ranker");
        assert_eq!(contract.version, "v1");
        assert_eq!(contract.data_sources[0].name, "user_profile_hbase");
        assert_eq!(contract.required_inputs[0].name, "user_id");
        assert_eq!(
            contract.required_inputs[0].data_source.as_deref(),
            Some("user_profile_hbase")
        );

        fs::remove_dir_all(root).unwrap();
    }

    fn write_lr_manifest_artifact(root: &Path, model_id: &str, version: &str) -> PathBuf {
        let version_dir = root.join(model_id).join(version);
        fs::create_dir_all(&version_dir).unwrap();
        let feature_config = version_dir.join("feature.yaml");
        let model_config = version_dir.join("model.yaml");
        let weights = version_dir.join("model.safetensors");
        let manifest_path = version_dir.join("model_manifest.yaml");

        let feature_yaml = "version: '1.0.0'\nsources:\n  - name: user_id\n    dtype: int\n    default_val: '0'\noperators:\n  - name: user_id_hash\n    op_type: FeatureHash\n    inputs: [user_id]\n    outputs: [user_id_idx]\n    params: {vocab_size: 100, num_hashes: 1}\n    embed: {vocab_size: 100, embed_dim: 8}\n";
        fs::write(&feature_config, feature_yaml).unwrap();
        fs::write(&model_config, "type: lr\n").unwrap();

        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((100, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &weights).unwrap();

        let manifest = format!(
            "schema_version: 1\nmodel_id: {}\nmodel_version: {}\nmodel_type: lr\ncode_commit:\nweights_file: model.safetensors\nweights_sha256: {}\nfeature_config_file: feature.yaml\nfeature_config_sha256: {}\nmodel_config_file: model.yaml\nmodel_config_sha256: {}\n",
            model_id,
            version,
            sha256_file(&weights).unwrap(),
            sha256_file(&feature_config).unwrap(),
            sha256_file(&model_config).unwrap(),
        );
        fs::write(&manifest_path, manifest).unwrap();
        manifest_path
    }

    #[test]
    fn supports_multiple_versions_default_and_fallback_resolution() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-multi-version-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let v1_manifest = write_lr_manifest_artifact(&root, "ranker", "20260604_010000");
        let v2_manifest = write_lr_manifest_artifact(&root, "ranker", "20260604_020000");

        let registry = ModelRegistry::from_model_dir(&root).unwrap();
        registry.load_manifest(&v1_manifest).unwrap();
        registry.load_manifest(&v2_manifest).unwrap();

        let default = registry.resolve("ranker", None, None).unwrap();
        assert_eq!(default.version, "20260604_020000");

        let explicit = registry
            .resolve("ranker", Some("20260604_010000"), None)
            .unwrap();
        assert_eq!(explicit.version, "20260604_010000");

        let fallback = registry
            .resolve("ranker", Some("missing"), Some("20260604_010000"))
            .unwrap();
        assert_eq!(fallback.version, "20260604_010000");
        assert!(registry.resolve("ranker", Some("missing"), None).is_none());

        let info = registry.model_info("ranker").unwrap();
        assert_eq!(info.name, "ranker");
        assert_eq!(info.default_version.as_deref(), Some("20260604_020000"));
        assert_eq!(info.versions.len(), 2);
        assert!(info
            .versions
            .iter()
            .any(|v| v.version == "20260604_020000" && v.is_default));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_aliases_and_weighted_routing_policies() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-alias-routing-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let v1_manifest = write_lr_manifest_artifact(&root, "ranker", "20260604_010000");
        let v2_manifest = write_lr_manifest_artifact(&root, "ranker", "20260604_020000");

        let registry = ModelRegistry::from_model_dir(&root).unwrap();
        registry.load_manifest(&v1_manifest).unwrap();
        registry.load_manifest(&v2_manifest).unwrap();

        registry
            .set_alias("ranker", "prod", "20260604_020000")
            .unwrap();
        let alias_resolved = registry
            .resolve_request("ranker", None, Some("prod"), None, None)
            .unwrap();
        assert_eq!(alias_resolved.version, "20260604_020000");

        registry
            .set_routing_policy(
                "ranker",
                Some(RoutingPolicy::Weighted {
                    key_field: Some("user_id".to_string()),
                    salt: Some("salt".to_string()),
                    versions: vec![
                        WeightedVersion {
                            version: "20260604_010000".to_string(),
                            weight: 0,
                        },
                        WeightedVersion {
                            version: "20260604_020000".to_string(),
                            weight: 100,
                        },
                    ],
                }),
            )
            .unwrap();
        let routed = registry
            .resolve_request("ranker", None, None, None, Some("user-42"))
            .unwrap();
        assert_eq!(routed.version, "20260604_020000");

        let info = registry.model_info("ranker").unwrap();
        assert_eq!(info.aliases.len(), 1);
        assert_eq!(info.aliases[0].alias, "prod");
        assert!(matches!(info.routing, Some(RoutingPolicy::Weighted { .. })));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_loose_safetensors_from_explicit_path() {
        let root = std::env::temp_dir().join(format!(
            "scale-rec-explicit-safetensors-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let feature_config = root.join("feature.yaml");
        let model_config = root.join("model_ranker.yaml");
        let weights = root.join("ranker.safetensors");

        let feature_yaml = "version: '1.0.0'\nsources:\n  - name: user_id\n    dtype: int\n    default_val: '0'\noperators:\n  - name: user_id_hash\n    op_type: FeatureHash\n    inputs: [user_id]\n    outputs: [user_id_idx]\n    params: {vocab_size: 100, num_hashes: 1}\n    embed: {vocab_size: 100, embed_dim: 8}\n";
        fs::write(&feature_config, feature_yaml).unwrap();
        fs::write(&model_config, "type: lr\n").unwrap();

        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((100, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, &weights).unwrap();

        let registry = ModelRegistry::new(&feature_config, &root).unwrap();
        let info = registry.load_safetensors(&weights).unwrap();

        assert_eq!(info.name, "ranker");
        assert_eq!(info.model_version.as_deref(), Some("default"));
        assert!(registry.get("ranker").is_some());

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
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
        );
        tensors1.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((100, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors1.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors1.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
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
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
        );
        tensors2.insert(
            "embeddings.emb_user_id_idx.weight".to_string(),
            Tensor::zeros((120, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors2.insert(
            "mlp.output.weight".to_string(),
            Tensor::zeros((1, 8), CandleDType::F32, &device).unwrap(),
        );
        tensors2.insert(
            "mlp.output.bias".to_string(),
            Tensor::zeros((1,), CandleDType::F32, &device).unwrap(),
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
