use std::collections::HashMap;
use std::time::Instant;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scale_rec_core::feats::builder::DagBuilder;
use scale_rec_core::feats::config::FlowConfig;
use scale_rec_core::feats::defaults::{parse_string_to_fv, source_default};
use scale_rec_core::feats::executor::DagExecutor;
use scale_rec_core::feats::feature_info::FeatureInfo;
use scale_rec_core::feats::ops::Fv;
use scale_rec_core::feats::tensor_utils::{feature_column_to_vec, FeatureColumn};
use scale_rec_core::feats::FeatureSpec;

/// 训练特征预处理引擎：加载 YAML 配置，构建 DAG，批量预处理 pandas 数据。
#[pyclass]
pub struct FeatSession {
    executor: DagExecutor,
    /// (FeatureSpec, column_id in execution context)
    embed_features: Vec<(FeatureSpec, usize)>,
}

#[pymethods]
impl FeatSession {
    #[new]
    fn new(config_path: &str) -> PyResult<Self> {
        let yaml = std::fs::read_to_string(config_path)
            .map_err(|e| PyIOError::new_err(format!("read config '{}': {}", config_path, e)))?;
        let flow_config = FlowConfig::from_yaml(&yaml)
            .map_err(|e| PyValueError::new_err(format!("parse config: {}", e)))?;
        let artifact = DagBuilder::build(flow_config)
            .map_err(|e| PyValueError::new_err(format!("build dag: {}", e)))?;

        let feat_info = FeatureInfo::new(
            artifact.sources.clone(),
            artifact.node_defs.clone(),
            artifact.execution_order.clone(),
        );

        let embed_ids = artifact.plan.embed_ids().to_vec();
        let embed_features: Vec<(FeatureSpec, usize)> = feat_info
            .embeddable_features()
            .into_iter()
            .zip(embed_ids.into_iter())
            .map(|((name, embed), col_id)| {
                let seq_len = embed.seq_len.or_else(|| {
                    artifact
                        .feature_schemas
                        .get(name)
                        .and_then(|schema| schema.dtype.list_len())
                });
                let spec = FeatureSpec {
                    name: name.to_string(),
                    vocab_size: embed.vocab_size,
                    embed_dim: embed.embed_dim,
                    pooling: embed.pooling,
                    seq_len,
                    truncation: embed.truncation,
                };
                (spec, col_id)
            })
            .collect();

        let executor = DagExecutor::new(
            artifact.plan,
            artifact.sources,
            artifact.execution_order,
            artifact.data_sources,
        );

        Ok(Self {
            executor,
            embed_features,
        })
    }

    /// 批量预处理：接收 pandas 列式数据，返回 {feat_name: list[int] | list[list[int]]}
    ///
    /// Args:
    ///     columns: dict[str, list[str | None]]
    ///         列名 → 每行的字符串值（None 表示缺失）
    ///
    /// Returns:
    ///     dict[str, list[int] | list[list[int]]]
    ///         只包含 embeddable features
    #[pyo3(signature = (columns))]
    fn preprocess_batch(
        &self,
        columns: HashMap<String, Vec<Option<String>>>,
        py: Python<'_>,
    ) -> PyResult<HashMap<String, PyObject>> {
        let (result, _timings) = self.preprocess_batch_inner(columns, py, false)?;
        Ok(result)
    }

    /// Profiled batch preprocessing. Returns (features, timings_seconds).
    #[pyo3(signature = (columns))]
    fn preprocess_batch_profile(
        &self,
        columns: HashMap<String, Vec<Option<String>>>,
        py: Python<'_>,
    ) -> PyResult<(HashMap<String, PyObject>, HashMap<String, f64>)> {
        self.preprocess_batch_inner(columns, py, true)
    }
}

impl FeatSession {
    fn preprocess_batch_inner(
        &self,
        columns: HashMap<String, Vec<Option<String>>>,
        py: Python<'_>,
        profile: bool,
    ) -> PyResult<(HashMap<String, PyObject>, HashMap<String, f64>)> {
        let total_start = Instant::now();
        let mut timings = HashMap::new();
        let n_rows = columns.values().next().map(|c| c.len()).unwrap_or(0);
        if n_rows == 0 {
            return Ok((HashMap::new(), timings));
        }

        // 1. parse strings to Fv
        let parse_start = Instant::now();
        let parsed = py
            .allow_threads(|| self.parse_columns(&columns))
            .map_err(|e| PyValueError::new_err(e))?;
        if profile {
            timings.insert(
                "rust_parse_s".to_string(),
                parse_start.elapsed().as_secs_f64(),
            );
        }

        // 2. execute DAG
        let skip_ops = std::collections::HashSet::new();
        let precomputed = HashMap::new();
        let execute_start = Instant::now();
        let context = if profile {
            let (context, op_timings) = py
                .allow_threads(|| {
                    self.executor
                        .execute_plan_with_timings(&parsed, &skip_ops, &precomputed)
                })
                .map_err(|e| PyRuntimeError::new_err(format!("dag execute: {}", e)))?;
            let mut op_type_totals: HashMap<String, f64> = HashMap::new();
            for timing in op_timings {
                timings.insert(format!("op:{}", timing.operator), timing.seconds);
                *op_type_totals.entry(timing.op_type).or_insert(0.0) += timing.seconds;
            }
            for (op_type, seconds) in op_type_totals {
                timings.insert(format!("op_type:{}", op_type), seconds);
            }
            context
        } else {
            py.allow_threads(|| self.executor.execute_plan(&parsed, &skip_ops, &precomputed))
                .map_err(|e| PyRuntimeError::new_err(format!("dag execute: {}", e)))?
        };
        if profile {
            timings.insert(
                "rust_execute_s".to_string(),
                execute_start.elapsed().as_secs_f64(),
            );
        }

        // 3. extract embed features
        let extract_start = Instant::now();
        let mut result = HashMap::new();
        for (spec, col_id) in &self.embed_features {
            if *col_id >= context.len() {
                continue;
            }
            let col = &context[*col_id];
            let fc = feature_column_to_vec(spec, col.as_slice(), n_rows)
                .map_err(|e| PyRuntimeError::new_err(format!("tensor '{}': {}", spec.name, e)))?;
            match fc {
                FeatureColumn::Scalar(v) => {
                    result.insert(spec.name.clone(), v.into_py(py));
                }
                FeatureColumn::Sequence(v) => {
                    result.insert(spec.name.clone(), v.into_py(py));
                }
            }
        }
        if profile {
            timings.insert(
                "rust_extract_s".to_string(),
                extract_start.elapsed().as_secs_f64(),
            );
            timings.insert(
                "rust_total_s".to_string(),
                total_start.elapsed().as_secs_f64(),
            );
        }

        Ok((result, timings))
    }

    fn parse_columns(
        &self,
        columns: &HashMap<String, Vec<Option<String>>>,
    ) -> Result<HashMap<String, Vec<Fv>>, String> {
        let source_defs = self.executor.source_defs();
        let mut parsed = HashMap::with_capacity(columns.len());

        for (name, values) in columns {
            let def = source_defs
                .get(name)
                .ok_or_else(|| format!("unknown source column '{}'", name))?;
            let default = source_default(def);
            let col: Vec<Fv> = values
                .iter()
                .map(|v| match v {
                    None => default.clone(),
                    Some(s) => {
                        parse_string_to_fv(s, &def.dtype).unwrap_or_else(|_| default.clone())
                    }
                })
                .collect();
            parsed.insert(name.clone(), col);
        }

        Ok(parsed)
    }
}

/// PyO3 module entry point.
#[pymodule]
fn feat_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FeatSession>()?;
    Ok(())
}
