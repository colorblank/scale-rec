//! FeatureDag facade —— 委派给 DagBuilder / DagExecutor / FeatureInfo。
//!
//! 新代码请直接使用 builder/executor/feature_info 模块，而非此 facade。

pub use super::builder::{DagArtifact, DagBuilder, ValidationIssue, ValidationReport};
pub use super::executor::{DagExecutor, ExecStep, ExecutionPlan};
use super::feature_info::{FeatureInfo, FeatureScope};
use super::ops::Fv;
use crate::feats::config::{DataSourceDef, FlowConfig, OperatorDef, SourceDef};
use crate::feats::debug::DebugTracer;
use crate::feats::defaults::source_default;
use crate::feats::schema::FeatureSchema;
use std::collections::{HashMap, HashSet};

/// 特征值的类型别名。
pub type FeatureValue = Fv;

/// 特征处理结果：包含所有特征值并区分来源。
pub struct FeatureResult {
    /// 所有 source 与计算结果的特征值。
    pub features: HashMap<String, FeatureValue>,
    /// 原始 source 名称集合。
    pub source_names: HashSet<String>,
    /// DAG 计算产生的特征名称集合。
    pub computed_names: HashSet<String>,
}

/// 特征 DAG 执行引擎（facade）。
///
/// 根据 FlowConfig 构建算子图，拓扑排序后按序执行单样本处理。
pub struct FeatureDag {
    sources: HashMap<String, SourceDef>,
    /// 配置中的外部数据源定义。
    pub data_sources: Vec<DataSourceDef>,
    node_defs: HashMap<String, OperatorDef>,
    /// 拓扑排序后的算子执行顺序。
    pub execution_order: Vec<String>,
    debug_mode: bool,
    /// 可选调试 tracer，用于记录执行过程。
    pub tracer: Option<DebugTracer>,
    /// 预编译执行计划。
    pub plan: ExecutionPlan,
    /// source 和算子输出的静态 schema。
    pub feature_schemas: HashMap<String, FeatureSchema>,
    /// DAG 构建校验报告。
    pub validation_report: ValidationReport,
}

impl FeatureDag {
    /// 从 FlowConfig 构建 FeatureDag，包含拓扑排序和预编译执行计划。
    pub fn from_config(
        config: FlowConfig,
        debug_mode: bool,
        tracer: Option<DebugTracer>,
    ) -> Result<Self, String> {
        let artifact = DagBuilder::build(config)?;
        let data_sources = artifact.data_sources;
        let sources = artifact.sources;
        let node_defs = artifact.node_defs;
        let execution_order = artifact.execution_order;
        let plan = artifact.plan;
        let feature_schemas = artifact.feature_schemas;
        let validation_report = artifact.validation_report;

        Ok(Self {
            sources,
            data_sources,
            node_defs,
            execution_order,
            debug_mode,
            tracer,
            plan,
            feature_schemas,
            validation_report,
        })
    }

    /// 获取原始输入源定义。
    pub fn source_defs(&self) -> &HashMap<String, SourceDef> {
        &self.sources
    }

    /// 获取算子节点定义。
    pub fn operator_defs(&self) -> &HashMap<String, OperatorDef> {
        &self.node_defs
    }

    /// 获取所有需要嵌入的特征 (name, EmbedConfig) 列表。
    pub fn embeddable_features(&self) -> Vec<(&str, &crate::feats::config::EmbedConfig)> {
        let mut result = Vec::new();
        for (_, op) in &self.node_defs {
            if let Some(ref emb) = op.embed {
                for out_name in &op.outputs {
                    result.push((out_name.as_str(), emb));
                }
            }
        }
        result.sort_by(|a, b| a.0.cmp(b.0));
        result
    }

    /// Classify each operator by user/item/context-derived feature scope.
    pub fn op_source_kind(&self) -> HashMap<String, FeatureScope> {
        FeatureInfo::new(
            self.sources.clone(),
            self.node_defs.clone(),
            self.feature_schemas.clone(),
            self.execution_order.clone(),
        )
        .op_source_kind()
    }

    /// Get source names (used to filter inputs in broadcast mode).
    pub fn source_names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }

    /// Get operator outputs by operator name.
    pub fn op_outputs(&self, op_name: &str) -> Option<&Vec<String>> {
        self.node_defs.get(op_name).map(|d| &d.outputs)
    }

    /// 执行 DAG：raw_inputs → 拓扑序执行算子 → FeatureResult。
    pub fn execute(
        &self,
        raw_inputs: &HashMap<String, FeatureValue>,
    ) -> Result<FeatureResult, String> {
        if let Some(ref tracer) = self.tracer {
            tracer.begin_sample();
        }

        let mut context: HashMap<String, FeatureValue> = HashMap::new();

        let mut overridden = Vec::new();
        for (name, val) in raw_inputs {
            if self.sources.contains_key(name) {
                context.insert(name.clone(), val.clone());
                overridden.push(name.clone());
            }
        }

        for (name, source_def) in &self.sources {
            if !context.contains_key(name) {
                let default_val = source_default(source_def);
                context.insert(name.clone(), default_val);
            }
        }

        let _ = &self.tracer;

        let source_names: HashSet<String> = self.sources.keys().cloned().collect();
        let mut computed_names = HashSet::new();
        for (op_idx, node_name) in self.execution_order.iter().enumerate() {
            let op = &self.plan.ops[op_idx];
            let def = &self.node_defs[node_name];
            let op_inputs: Vec<Fv> = def
                .inputs
                .iter()
                .map(|inp| {
                    context
                        .get(inp)
                        .map(|v| v.clone())
                        .ok_or_else(|| format!("Required input '{}' not found", inp))
                })
                .collect::<Result<_, String>>()?;
            let output = op.process(&op_inputs)?;
            let output_fv: FeatureValue = output;
            for out_name in &def.outputs {
                context.insert(out_name.clone(), output_fv.clone());
                computed_names.insert(out_name.clone());
            }
        }
        if self.debug_mode {
            self.dump_snapshot(&context, &source_names, &computed_names);
        }
        if let Some(ref tracer) = self.tracer {
            tracer.end_sample();
        }
        Ok(FeatureResult {
            features: context,
            source_names,
            computed_names,
        })
    }

    /// Batch DAG execution on columnar data.
    pub fn execute_batch(
        &self,
        columns: &HashMap<String, Vec<FeatureValue>>,
        skip_ops: &HashSet<String>,
    ) -> Result<HashMap<String, Vec<FeatureValue>>, String> {
        self.execute_batch_precomputed(columns, skip_ops, &HashMap::new())
    }

    /// Batch execution with precomputed single-value features broadcast to all rows.
    pub fn execute_batch_precomputed(
        &self,
        columns: &HashMap<String, Vec<FeatureValue>>,
        skip_ops: &HashSet<String>,
        precomputed: &HashMap<String, FeatureValue>,
    ) -> Result<HashMap<String, Vec<FeatureValue>>, String> {
        let n_rows = columns.values().next().map(|v| v.len()).unwrap_or(0);
        if n_rows == 0 {
            return Ok(HashMap::new());
        }

        let mut context: HashMap<String, Vec<FeatureValue>> = HashMap::new();

        for (name, col) in columns {
            if self.sources.contains_key(name) {
                context.insert(name.clone(), col.clone());
            }
        }
        for (name, source_def) in &self.sources {
            if !context.contains_key(name) {
                let default = source_default(source_def);
                context.insert(name.clone(), vec![default; n_rows]);
            }
        }
        for (name, val) in precomputed {
            context.insert(name.clone(), vec![val.clone(); n_rows]);
        }

        let default_col: Vec<FeatureValue> = vec![Fv::Int(0); n_rows];
        for (op_idx, node_name) in self.execution_order.iter().enumerate() {
            if skip_ops.contains(node_name) {
                continue;
            }
            let op = &self.plan.ops[op_idx];
            let def = &self.node_defs[node_name];

            let input_slices: Vec<&[FeatureValue]> = def
                .inputs
                .iter()
                .map(|inp| {
                    context
                        .get(inp)
                        .map(|c| c.as_slice())
                        .unwrap_or(default_col.as_slice())
                })
                .collect();

            let result_vec = op
                .process_batch(&input_slices, n_rows)
                .map_err(|e| format!("{}: {}", node_name, e))?;

            for out_name in &def.outputs {
                context.insert(out_name.clone(), result_vec.clone());
            }
        }

        Ok(context)
    }

    /// 调试用：输出当前 context 的快照日志。
    pub fn dump_snapshot(
        &self,
        context: &HashMap<String, FeatureValue>,
        source_names: &HashSet<String>,
        computed_names: &HashSet<String>,
    ) {
        tracing::info!("Feature Snapshot");
        for (name, val) in context {
            let type_name = val.type_name();
            let origin = if computed_names.contains(name) {
                "computed"
            } else if source_names.contains(name) {
                "source"
            } else {
                "raw"
            };
            tracing::info!(origin, name, type_name, "feature snapshot entry");
        }
    }
}
