//! DAG 构建器：解析 FlowConfig，构建 DAG 拓扑、校验、预编译执行计划。
use std::collections::{HashMap, HashSet};

use petgraph::algo::toposort;
use petgraph::prelude::DiGraph;

use super::ops::registry;
use super::ops::{CustomOp, Fv};
use crate::feats::config::{FlowConfig, OperatorDef, Role, SourceDef};
use crate::feats::defaults::source_default;
use crate::feats::executor::{ExecStep, ExecutionPlan};
use crate::feats::schema::{infer_feature_schemas, FeatureSchema};

/// DAG 校验问题：包含严重级别、错误码和描述。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub severity: &'static str,
    pub code: &'static str,
    pub message: String,
    pub feature: Option<String>,
}

/// DAG 校验报告：汇总 source 消费率、embed 利用率和中间结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
    pub source_count: usize,
    pub embeddable_count: usize,
    pub intermediate_count: usize,
}

impl ValidationReport {
    pub fn warnings(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == "warning")
    }

    pub fn errors(&self) -> impl Iterator<Item = &ValidationIssue> {
        self.issues.iter().filter(|issue| issue.severity == "error")
    }
}

/// DAG 构建产物：供 DagExecutor 和 FeatureInfo 消费。
pub struct DagArtifact {
    pub sources: HashMap<String, SourceDef>,
    pub node_defs: HashMap<String, OperatorDef>,
    pub execution_order: Vec<String>,
    pub plan: ExecutionPlan,
    pub feature_schemas: HashMap<String, FeatureSchema>,
    pub data_sources: Vec<crate::feats::config::DataSourceDef>,
    pub validation_report: ValidationReport,
}

/// DAG 构建器：解析 FlowConfig → 拓扑排序 → 校验 → 预编译 ExecutionPlan。
pub struct DagBuilder;

impl DagBuilder {
    /// 从 FlowConfig 构建 DAG 并预编译执行计划。
    pub fn build(config: FlowConfig) -> Result<DagArtifact, String> {
        let feature_schemas = infer_feature_schemas(&config.sources, &config.operators)?;

        let data_source_names: HashSet<String> = config
            .data_sources
            .iter()
            .map(|source| source.name.clone())
            .collect();
        for source in &config.sources {
            if let Some(data_source) = &source.data_source {
                if !data_source_names.contains(data_source) {
                    return Err(format!(
                        "source '{}' references unknown data_source '{}'",
                        source.name, data_source
                    ));
                }
            }
        }
        let data_sources = config.data_sources.clone();

        let mut sources = HashMap::new();
        for s in config.sources {
            if s.role != Role::Feature {
                tracing::info!("source '{}' skipped (role={:?})", s.name, s.role);
                continue;
            }
            if sources.contains_key(&s.name) {
                return Err(format!("Duplicate source name: '{}'", s.name));
            }
            sources.insert(s.name.clone(), s);
        }
        let mut valid_inputs: HashSet<String> = sources.keys().cloned().collect();
        let mut nodes = HashMap::new();
        let mut node_defs = HashMap::new();
        let mut graph = DiGraph::<String, ()>::new();
        let mut name_to_idx = HashMap::new();
        for op_def in &config.operators {
            if nodes.contains_key(&op_def.name) {
                return Err(format!("Duplicate operator name: '{}'", op_def.name));
            }
            let op = Self::create_op(op_def)?;
            nodes.insert(op_def.name.clone(), op);
            node_defs.insert(op_def.name.clone(), op_def.clone());
            let idx = graph.add_node(op_def.name.clone());
            name_to_idx.insert(op_def.name.clone(), idx);
        }
        let mut output_to_provider = HashMap::new();
        for op_def in &config.operators {
            for out in &op_def.outputs {
                if valid_inputs.contains(out) {
                    return Err(format!("Duplicate output name '{}'", out));
                }
                valid_inputs.insert(out.clone());
                output_to_provider.insert(out.clone(), op_def.name.clone());
            }
        }
        for op_def in &config.operators {
            for input_name in &op_def.inputs {
                if !valid_inputs.contains(input_name) {
                    return Err(format!(
                        "Operator '{}' references unknown input '{}'",
                        op_def.name, input_name
                    ));
                }
            }
        }
        for op_def in &config.operators {
            let target_idx = name_to_idx[&op_def.name];
            for input_name in &op_def.inputs {
                if let Some(provider_name) = output_to_provider.get(input_name) {
                    let source_idx = name_to_idx[provider_name];
                    graph.add_edge(source_idx, target_idx, ());
                }
            }
        }
        let sorted_indices =
            toposort(&graph, None).map_err(|_| "Cycle detected in feature DAG".to_string())?;
        let execution_order: Vec<String> = sorted_indices
            .iter()
            .map(|&idx| graph[idx].clone())
            .collect();

        // Build precompiled execution plan
        let mut col_id: HashMap<String, usize> = HashMap::new();
        let mut source_cols: Vec<usize> = Vec::new();
        let mut source_names: Vec<String> = Vec::new();
        let mut source_defaults: Vec<Fv> = Vec::new();
        let mut col_names: Vec<Option<String>> = vec![None; sources.len()];
        for (i, s) in sources.keys().enumerate() {
            col_id.insert(s.clone(), i);
            source_cols.push(i);
            source_names.push(s.clone());
            col_names[i] = Some(s.clone());
            let source_def = sources
                .get(s)
                .ok_or_else(|| format!("source '{}' missing while building execution plan", s))?;
            source_defaults.push(source_default(source_def));
        }
        let mut col_count = sources.len();
        for op_def in &config.operators {
            for out in &op_def.outputs {
                if !col_id.contains_key(out) {
                    col_id.insert(out.clone(), col_count);
                    col_names.push(Some(out.clone()));
                    col_count += 1;
                }
            }
        }
        let mut plan_steps: Vec<ExecStep> = Vec::with_capacity(execution_order.len());
        let mut plan_ops: Vec<Box<dyn CustomOp>> = Vec::with_capacity(nodes.len());
        let mut op_name_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, node_name) in execution_order.iter().enumerate() {
            op_name_to_idx.insert(node_name.clone(), i);
            plan_ops.push(nodes.remove(node_name).ok_or_else(|| {
                format!("operator '{}' missing after topological sort", node_name)
            })?);
        }
        for node_name in &execution_order {
            let def = &node_defs[node_name];
            let op_idx = op_name_to_idx[node_name];
            let input_cols: Vec<usize> = def
                .inputs
                .iter()
                .map(|inp| {
                    col_id
                        .get(inp)
                        .copied()
                        .ok_or_else(|| format!("input column '{}' missing from plan", inp))
                })
                .collect::<Result<_, _>>()?;
            let output_cols: Vec<usize> = def
                .outputs
                .iter()
                .map(|out| {
                    col_id
                        .get(out)
                        .copied()
                        .ok_or_else(|| format!("output column '{}' missing from plan", out))
                })
                .collect::<Result<_, _>>()?;
            plan_steps.push(ExecStep {
                op_idx,
                input_cols,
                output_cols,
            });
        }
        let mut embed_pairs: Vec<(&str, usize)> = Vec::new();
        for op_def in &config.operators {
            if op_def.embed.is_some() {
                for out_name in &op_def.outputs {
                    if let Some(&id) = col_id.get(out_name) {
                        embed_pairs.push((out_name, id));
                    }
                }
            }
        }
        embed_pairs.sort_by_key(|(name, _)| *name);

        let validation_report = Self::validate(&sources, &config.operators, &embed_pairs);

        let embed_ids: Vec<usize> = embed_pairs.into_iter().map(|(_, id)| id).collect();
        let plan = ExecutionPlan::new(
            plan_steps,
            plan_ops,
            source_cols,
            source_names,
            col_names,
            source_defaults,
            col_count,
            embed_ids,
        );

        Ok(DagArtifact {
            sources,
            node_defs,
            execution_order,
            plan,
            feature_schemas,
            data_sources,
            validation_report,
        })
    }

    fn create_op(def: &OperatorDef) -> Result<Box<dyn CustomOp>, String> {
        registry::create_op(&def.op_type, &def.params)
    }

    fn validate(
        sources: &HashMap<String, SourceDef>,
        operators: &[OperatorDef],
        embed_pairs: &[(&str, usize)],
    ) -> ValidationReport {
        let embeddable: HashSet<&str> = embed_pairs.iter().map(|(n, _)| *n).collect();
        let mut downstream: HashSet<&str> = HashSet::new();
        for op in operators {
            for inp in &op.inputs {
                downstream.insert(inp.as_str());
            }
        }

        let orphan_src: Vec<&str> = sources
            .keys()
            .filter(|n| !downstream.contains(n.as_str()) && !embeddable.contains(n.as_str()))
            .map(|s| s.as_str())
            .collect();

        let mut orphan_out: Vec<(&str, &str)> = Vec::new();
        let mut intermediate = 0usize;
        for op in operators {
            for out in &op.outputs {
                if embeddable.contains(out.as_str()) {
                    continue;
                }
                if downstream.contains(out.as_str()) {
                    intermediate += 1;
                    continue;
                }
                orphan_out.push((op.name.as_str(), out.as_str()));
            }
        }

        let mut issues = Vec::new();
        for source in &orphan_src {
            issues.push(ValidationIssue {
                severity: "warning",
                code: "orphan_source",
                message: format!("source '{}' is not consumed and not embeddable", source),
                feature: Some((*source).to_string()),
            });
        }
        for (op_name, output) in &orphan_out {
            issues.push(ValidationIssue {
                severity: "warning",
                code: "orphan_output",
                message: format!(
                    "operator '{}' output '{}' has no consumer and no embed",
                    op_name, output
                ),
                feature: Some((*output).to_string()),
            });
        }

        tracing::info!(
            consumed = sources.len() - orphan_src.len(),
            total = sources.len(),
            orphan = orphan_src.len(),
            "DAG source validation summary"
        );
        tracing::info!(
            embeddable = embeddable.len(),
            intermediate,
            orphan = orphan_out.len(),
            "DAG output validation summary"
        );
        if !orphan_src.is_empty() {
            tracing::warn!(?orphan_src, "DAG orphan sources");
        }
        if !orphan_out.is_empty() {
            tracing::warn!(
                count = orphan_out.len(),
                "DAG orphan outputs have no consumer and no embed"
            );
        }
        ValidationReport {
            issues,
            source_count: sources.len(),
            embeddable_count: embeddable.len(),
            intermediate_count: intermediate,
        }
    }
}
