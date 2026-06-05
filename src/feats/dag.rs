//! 特征 DAG 执行器：拓扑排序、算子调度、单样本执行。
use super::ops::{
    Bucketing, ConcatHash, CrossFeature, CustomOp, DictMapper, ExpressionOp, FeatureHash,
    FlatSplit, Fv, JsonExtractList, ListOverlap, ListStringParser, ParsedFeatureHash, PluginOp,
    SequenceOp, Split, StringConcat, StringParser,
};
use crate::feats::config::{FlowConfig, OperatorDef, SourceDef};
use crate::feats::debug::DebugTracer;
use crate::feats::defaults::parse_default;
use crate::feats::schema::{infer_feature_schemas, FeatureSchema};
use petgraph::algo::toposort;
use petgraph::prelude::DiGraph;
use std::collections::{HashMap, HashSet};

pub type FeatureValue = Fv;

/// 特征处理结果：包含所有特征值并区分来源。
pub struct FeatureResult {
    pub features: HashMap<String, FeatureValue>,
    pub source_names: HashSet<String>,
    pub computed_names: HashSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub severity: &'static str,
    pub code: &'static str,
    pub message: String,
    pub feature: Option<String>,
}

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

/// 预编译执行计划：运算符 + 整数索引列，运行时零 HashMap 查找。
pub struct ExecutionPlan {
    pub steps: Vec<ExecStep>,
    pub ops: Vec<Box<dyn CustomOp>>,
    source_cols: Vec<usize>,
    source_names: Vec<String>,
    col_names: Vec<Option<String>>,
    source_defaults: Vec<Fv>,
    col_count: usize,
    embed_ids: Vec<usize>,
}

pub struct ExecStep {
    pub op_idx: usize,
    pub input_cols: Vec<usize>,
    pub output_cols: Vec<usize>,
}

/// 特征 DAG 执行引擎。
///
/// 根据 FlowConfig 构建算子图，拓扑排序后按序执行单样本处理。
pub struct FeatureDag {
    sources: HashMap<String, SourceDef>,
    node_defs: HashMap<String, OperatorDef>,
    pub execution_order: Vec<String>,
    debug_mode: bool,
    pub tracer: Option<DebugTracer>,
    pub plan: ExecutionPlan,
    pub feature_schemas: HashMap<String, FeatureSchema>,
    pub validation_report: ValidationReport,
}

impl FeatureDag {
    pub fn from_config(
        config: FlowConfig,
        debug_mode: bool,
        tracer: Option<DebugTracer>,
    ) -> Result<Self, String> {
        use crate::feats::config::Role;

        let feature_schemas = infer_feature_schemas(&config.sources, &config.operators)?;

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

        // Build precompiled execution plan (column IDs for zero-HashMap execution)
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
            let source_def = sources.get(s).unwrap();
            source_defaults.push(parse_default(&source_def.default_val, &source_def.dtype));
        }
        let mut col_count = sources.len();
        // Assign IDs to operator outputs
        for op_def in &config.operators {
            for out in &op_def.outputs {
                if !col_id.contains_key(out) {
                    col_id.insert(out.clone(), col_count);
                    col_names.push(Some(out.clone()));
                    col_count += 1;
                }
            }
        }
        // Build steps with pre-resolved column indices
        let mut plan_steps: Vec<ExecStep> = Vec::with_capacity(execution_order.len());
        let mut plan_ops: Vec<Box<dyn CustomOp>> = Vec::with_capacity(nodes.len());
        let mut op_name_to_idx: HashMap<String, usize> = HashMap::new();
        for (i, node_name) in execution_order.iter().enumerate() {
            op_name_to_idx.insert(node_name.clone(), i);
            plan_ops.push(nodes.remove(node_name).unwrap());
        }
        for node_name in &execution_order {
            let def = &node_defs[node_name];
            let op_idx = op_name_to_idx[node_name];
            let input_cols: Vec<usize> = def
                .inputs
                .iter()
                .map(|inp| *col_id.get(inp).unwrap_or(&usize::MAX))
                .collect();
            let output_cols: Vec<usize> = def
                .outputs
                .iter()
                .map(|out| *col_id.get(out).unwrap_or(&usize::MAX))
                .collect();
            plan_steps.push(ExecStep {
                op_idx,
                input_cols,
                output_cols,
            });
        }
        // Embeddable column IDs — 全部来自算子 embed 配置
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
        let plan = ExecutionPlan {
            steps: plan_steps,
            ops: plan_ops,
            source_cols,
            source_names,
            col_names,
            source_defaults,
            col_count,
            embed_ids,
        };

        Ok(Self {
            sources,
            node_defs,
            execution_order,
            debug_mode,
            tracer,
            plan,
            feature_schemas,
            validation_report,
        })
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

        println!(
            "[DAG] sources: consumed={}/{} orphan={}",
            sources.len() - orphan_src.len(),
            sources.len(),
            orphan_src.len()
        );
        println!(
            "[DAG] outputs: embeddable={} intermediate={} orphan={}",
            embeddable.len(),
            intermediate,
            orphan_out.len()
        );
        if !orphan_src.is_empty() {
            println!("[DAG] WARNING orphan sources: {:?}", orphan_src);
        }
        if !orphan_out.is_empty() {
            println!(
                "[DAG] WARNING {} orphan outputs (no consumer, no embed)",
                orphan_out.len()
            );
        }
        ValidationReport {
            issues,
            source_count: sources.len(),
            embeddable_count: embeddable.len(),
            intermediate_count: intermediate,
        }
    }

    fn yaml_get<'a>(params: &'a serde_yaml::Value, key: &str) -> Option<&'a serde_yaml::Value> {
        params
            .as_mapping()?
            .get(&serde_yaml::Value::String(key.to_string()))
    }
    fn yaml_str<'a>(params: &'a serde_yaml::Value, key: &str) -> Option<&'a str> {
        Self::yaml_get(params, key)?.as_str()
    }
    fn yaml_i64(params: &serde_yaml::Value, key: &str) -> Option<i64> {
        Self::yaml_get(params, key)?.as_i64()
    }
    fn yaml_stringish(params: &serde_yaml::Value, key: &str) -> Option<String> {
        let value = Self::yaml_get(params, key)?;
        value
            .as_str()
            .map(ToOwned::to_owned)
            .or_else(|| value.as_i64().map(|v| v.to_string()))
            .or_else(|| value.as_u64().map(|v| v.to_string()))
    }
    fn yaml_f32_seq(params: &serde_yaml::Value, key: &str) -> Vec<f32> {
        Self::yaml_get(params, key)
            .and_then(|v| v.as_sequence())
            .map(|seq| {
                seq.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn create_op(def: &OperatorDef) -> Result<Box<dyn CustomOp>, String> {
        let p = &def.params;
        match def.op_type.as_str() {
            "DictMapper" => {
                let default_idx = Self::yaml_i64(p, "default_idx").unwrap_or(0) as i32;
                let mut mapping = HashMap::new();
                if let Some(map) = Self::yaml_get(p, "mapping").and_then(|v| v.as_mapping()) {
                    for (k, v) in map {
                        if let (Some(key), Some(val)) = (k.as_str(), v.as_i64()) {
                            mapping.insert(key.to_string(), val as i32);
                        }
                    }
                }
                Ok(Box::new(DictMapper::new(mapping, default_idx)))
            }
            "StringParser" => {
                let sep1 = Self::yaml_str(p, "sep1").unwrap_or("#").to_string();
                let sep2 = Self::yaml_str(p, "sep2").unwrap_or("|").to_string();
                let key_index = Self::yaml_i64(p, "key_index").unwrap_or(0) as usize;
                let pad_len = Self::yaml_i64(p, "pad_len").unwrap_or(0) as usize;
                let pad_val = Self::yaml_str(p, "pad_val")
                    .unwrap_or("unknown")
                    .to_string();
                Ok(Box::new(StringParser::new(
                    sep1, sep2, key_index, pad_len, pad_val,
                )))
            }
            "JsonExtractList" => {
                let key = Self::yaml_str(p, "key").map(|s| s.to_string());
                let pad_len = Self::yaml_i64(p, "pad_len").unwrap_or(0) as usize;
                let pad_val = Self::yaml_str(p, "pad_val").unwrap_or("").to_string();
                Ok(Box::new(JsonExtractList::new(key, pad_len, pad_val)))
            }
            "ListStringParser" => {
                let sep = Self::yaml_str(p, "sep").unwrap_or(",").to_string();
                let key_index = Self::yaml_i64(p, "key_index").unwrap_or(0) as usize;
                Ok(Box::new(ListStringParser::new(sep, key_index)))
            }
            "CrossFeature" => {
                let cross_type = Self::yaml_str(p, "cross_type")
                    .unwrap_or("cartesian")
                    .to_string();
                Ok(Box::new(CrossFeature::new(cross_type)))
            }
            "Bucketing" => {
                let boundaries = Self::yaml_f32_seq(p, "boundaries");
                Ok(Box::new(Bucketing::new(boundaries)))
            }
            "SequenceOp" => {
                let max_len = Self::yaml_i64(p, "max_len").unwrap_or(10) as usize;
                let pad_val = Self::yaml_i64(p, "pad_val").unwrap_or(0) as i32;
                Ok(Box::new(SequenceOp::new(max_len, pad_val)))
            }
            "Split" => {
                let sep = Self::yaml_str(p, "sep").unwrap_or("|").to_string();
                let max_len = Self::yaml_i64(p, "max_len").unwrap_or(0) as usize;
                let pad_val = Self::yaml_str(p, "pad_val").unwrap_or("").to_string();
                Ok(Box::new(Split::new(sep, max_len, pad_val)))
            }
            "FlatSplit" => {
                let sep = Self::yaml_str(p, "sep").unwrap_or(",").to_string();
                let max_len = Self::yaml_i64(p, "max_len").unwrap_or(0) as usize;
                let pad_val = Self::yaml_str(p, "pad_val").unwrap_or("").to_string();
                Ok(Box::new(FlatSplit::new(sep, max_len, pad_val)))
            }
            "ExpressionOp" => {
                let script = Self::yaml_str(p, "script")
                    .ok_or("Missing script for ExpressionOp")?
                    .to_string();
                Ok(Box::new(ExpressionOp::new(script)))
            }
            "PluginOp" => {
                let path = Self::yaml_str(p, "path")
                    .ok_or("Missing path for PluginOp")?
                    .to_string();
                let op_name = Self::yaml_str(p, "op_name")
                    .unwrap_or("custom_plugin")
                    .to_string();
                Ok(Box::new(PluginOp::new(&path, op_name)?))
            }
            "ListOverlap" => Ok(Box::new(ListOverlap::new())),
            "StringConcat" => {
                let separator = Self::yaml_str(p, "separator").unwrap_or("_").to_string();
                Ok(Box::new(StringConcat::new(separator)))
            }
            "FeatureHash" => {
                let vocab_size = Self::yaml_i64(p, "vocab_size").unwrap_or(1000) as u32;
                let num_hashes = Self::yaml_i64(p, "num_hashes").unwrap_or(1) as u32;
                let separator = Self::yaml_str(p, "separator").unwrap_or("|").to_string();
                let namespace = Self::yaml_stringish(p, "namespace").unwrap_or_default();
                let salt = Self::yaml_stringish(p, "salt").unwrap_or_default();
                let version = Self::yaml_stringish(p, "version").unwrap_or_default();
                Ok(Box::new(FeatureHash::with_scope(
                    vocab_size, num_hashes, separator, &namespace, &salt, &version,
                )))
            }
            "ParsedFeatureHash" => {
                let vocab_size = Self::yaml_i64(p, "vocab_size").unwrap_or(1000) as u32;
                let parse_mode = Self::yaml_str(p, "parse_mode")
                    .unwrap_or("json")
                    .to_string();
                let num_hashes = Self::yaml_i64(p, "num_hashes").unwrap_or(1) as u32;
                let separator = Self::yaml_str(p, "separator").unwrap_or("|").to_string();
                let namespace = Self::yaml_stringish(p, "namespace").unwrap_or_default();
                let salt = Self::yaml_stringish(p, "salt").unwrap_or_default();
                let version = Self::yaml_stringish(p, "version").unwrap_or_default();
                let key = Self::yaml_str(p, "key").map(|s| s.to_string());
                let sep1 = Self::yaml_str(p, "sep1").unwrap_or("|").to_string();
                let sep2 = Self::yaml_str(p, "sep2").unwrap_or("#").to_string();
                let key_index = Self::yaml_i64(p, "key_index").unwrap_or(0) as usize;
                let sep = Self::yaml_str(p, "sep").unwrap_or(",").to_string();
                let max_len = Self::yaml_i64(p, "max_len").unwrap_or(0) as usize;
                let pad_len = Self::yaml_i64(p, "pad_len").unwrap_or(0) as usize;
                let pad_val = Self::yaml_str(p, "pad_val").unwrap_or("").to_string();
                Ok(Box::new(ParsedFeatureHash::new(
                    vocab_size, parse_mode, num_hashes, separator, namespace, salt, version, key,
                    sep1, sep2, key_index, sep, max_len, pad_len, pad_val,
                )))
            }
            "ConcatHash" => {
                let vocab_size = Self::yaml_i64(p, "vocab_size").unwrap_or(1000) as u32;
                let num_hashes = Self::yaml_i64(p, "num_hashes").unwrap_or(1) as u32;
                let separator = Self::yaml_str(p, "separator").unwrap_or("_").to_string();
                let namespace = Self::yaml_stringish(p, "namespace").unwrap_or_default();
                let salt = Self::yaml_stringish(p, "salt").unwrap_or_default();
                let version = Self::yaml_stringish(p, "version").unwrap_or_default();
                Ok(Box::new(ConcatHash::new(
                    vocab_size, num_hashes, separator, namespace, salt, version,
                )))
            }
            _ => Err(format!("Unsupported operator type: {}", def.op_type)),
        }
    }

    pub fn source_defs(&self) -> &HashMap<String, SourceDef> {
        &self.sources
    }
    pub fn operator_defs(&self) -> &HashMap<String, OperatorDef> {
        &self.node_defs
    }

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

    /// Classify each operator by input source: "user", "item", or "cross" (both).
    pub fn op_source_kind(&self) -> HashMap<String, &str> {
        // Classify sources by their `source` field (User/Context→broadcast, Item/ItemStats→item)
        let mut feat_kind: HashMap<String, &str> = HashMap::new();
        for (name, src) in &self.sources {
            let k = match src.source.as_deref() {
                Some("User") | Some("Context") => "user",
                _ => "item", // Item, ItemStats, None, and anything else
            };
            feat_kind.insert(name.clone(), k);
        }
        // Propagate through operators in topological order
        for node_name in &self.execution_order {
            let def = &self.node_defs[node_name];
            let kinds: Vec<&&str> = def
                .inputs
                .iter()
                .filter_map(|inp| feat_kind.get(inp))
                .collect();
            let k = if kinds.iter().any(|k| **k == "user") && kinds.iter().any(|k| **k == "item") {
                "cross"
            } else if kinds.iter().any(|k| **k == "user") {
                "user"
            } else if kinds.iter().any(|k| **k == "item") {
                "item"
            } else {
                kinds.first().copied().unwrap_or(&&"other")
            };
            for out_name in &def.outputs {
                feat_kind.insert(out_name.clone(), k);
            }
        }
        // Map operator name → kind
        let mut op_kind: HashMap<String, &str> = HashMap::new();
        for node_name in &self.execution_order {
            let def = &self.node_defs[node_name];
            let kinds: Vec<&&str> = def
                .inputs
                .iter()
                .filter_map(|inp| feat_kind.get(inp))
                .collect();
            let k = if kinds.iter().any(|k| **k == "user") && kinds.iter().any(|k| **k == "item") {
                "cross"
            } else if kinds.iter().any(|k| **k == "user") {
                "user"
            } else if kinds.iter().any(|k| **k == "item") {
                "item"
            } else {
                "other"
            };
            op_kind.insert(node_name.clone(), k);
        }
        op_kind
    }

    /// Get source names (used to filter inputs in broadcast mode).
    pub fn source_names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }
    /// Get operator outputs by operator name.
    pub fn op_outputs(&self, op_name: &str) -> Option<&Vec<String>> {
        self.node_defs.get(op_name).map(|d| &d.outputs)
    }

    pub fn execute(
        &self,
        raw_inputs: &HashMap<String, FeatureValue>,
    ) -> Result<FeatureResult, String> {
        if let Some(ref tracer) = self.tracer {
            tracer.begin_sample();
        }

        let mut context: HashMap<String, FeatureValue> = HashMap::new();

        // Stage 1: raw inputs first (avoid allocating defaults that will be overwritten)
        let mut overridden = Vec::new();
        for (name, val) in raw_inputs {
            if self.sources.contains_key(name) {
                context.insert(name.clone(), val.clone());
                overridden.push(name.clone());
            }
        }

        // Stage 2: fill defaults only for missing keys
        for (name, source_def) in &self.sources {
            if !context.contains_key(name) {
                let default_val = parse_default(&source_def.default_val, &source_def.dtype);
                context.insert(name.clone(), default_val);
            }
        }

        // tracer disabled — needs update for Fv enum
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

    /// Batch DAG execution on columnar data. Operators with `process_batch()` are
    /// invoked once per column, eliminating per-row trait object dispatch overhead.
    pub fn execute_batch(
        &self,
        columns: &HashMap<String, Vec<FeatureValue>>,
        skip_ops: &HashSet<String>,
    ) -> Result<HashMap<String, Vec<FeatureValue>>, String> {
        self.execute_batch_precomputed(columns, skip_ops, &HashMap::new())
    }

    /// Batch execution with precomputed single-value features broadcast to all rows.
    /// Used for broadcast mode: user features are precomputed once, then injected.
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

        // Stage 1: raw inputs
        for (name, col) in columns {
            if self.sources.contains_key(name) {
                context.insert(name.clone(), col.clone());
            }
        }
        // Stage 2: fill defaults for missing keys
        for (name, source_def) in &self.sources {
            if !context.contains_key(name) {
                let default = parse_default(&source_def.default_val, &source_def.dtype);
                context.insert(name.clone(), vec![default; n_rows]);
            }
        }
        // Stage 2.5: inject precomputed values (broadcast single value to N rows)
        for (name, val) in precomputed {
            context.insert(name.clone(), vec![val.clone(); n_rows]);
        }

        // Stage 3: execute operators in topological order (bulk, zero-copy refs)
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

    pub fn dump_snapshot(
        &self,
        context: &HashMap<String, FeatureValue>,
        source_names: &HashSet<String>,
        computed_names: &HashSet<String>,
    ) {
        println!("[Feature Snapshot]");
        for (name, val) in context {
            let type_name = val.type_name();
            let origin = if computed_names.contains(name) {
                "computed"
            } else if source_names.contains(name) {
                "source"
            } else {
                "raw"
            };
            println!(" -> [{:<8}] {:<20} | Type: {}", origin, name, type_name);
        }
    }
}

impl ExecutionPlan {
    /// Fast execution using pre-compiled column indices. Zero HashMap lookups.
    pub fn execute_plan(
        &self,
        columns: &HashMap<String, Vec<Fv>>,
        skip_op_idx: &HashSet<usize>,
        precomputed: &HashMap<usize, Fv>,
    ) -> Result<Vec<Vec<Fv>>, String> {
        let n_rows = columns.values().next().map(|v| v.len()).unwrap_or(0);
        if n_rows == 0 {
            return Ok(Vec::new());
        }

        let mut context: Vec<Vec<Fv>> = vec![Vec::with_capacity(n_rows); self.col_count];

        // Stage 1: fill sources from input
        for i in 0..self.source_cols.len() {
            let name = &self.source_names[i];
            let cid = self.source_cols[i];
            let source_default = self.source_defaults.get(i).cloned().unwrap_or(Fv::Int(0));
            if let Some(col) = columns.get(name) {
                if col.len() == n_rows {
                    context[cid] = col.clone();
                } else {
                    let mut fixed = vec![source_default; n_rows];
                    for (row, val) in col.iter().take(n_rows).enumerate() {
                        fixed[row] = val.clone();
                    }
                    context[cid] = fixed;
                }
            } else {
                context[cid] = vec![source_default; n_rows];
            }
        }
        // Stage 2: inject precomputed (broadcast mode)
        for (&col_id, val) in precomputed.iter() {
            if col_id < context.len() {
                context[col_id] = vec![val.clone(); n_rows];
            }
        }

        // Stage 3: execute steps in order
        for step in &self.steps {
            if skip_op_idx.contains(&step.op_idx) {
                continue;
            }
            let op = &self.ops[step.op_idx];
            let input_slices: Vec<&[Fv]> = step
                .input_cols
                .iter()
                .map(|&cid| {
                    context.get(cid).map(|c| c.as_slice()).ok_or_else(|| {
                        let name = self
                            .col_names
                            .get(cid)
                            .and_then(|n| n.as_deref())
                            .unwrap_or("<unknown>");
                        format!("missing required column '{}' (id={})", name, cid)
                    })
                })
                .collect::<Result<_, String>>()?;
            let result_vec = op
                .process_batch(&input_slices, n_rows)
                .map_err(|e| format!("step {}: {}", step.op_idx, e))?;
            for &cid in &step.output_cols {
                context[cid] = result_vec.clone();
            }
        }
        Ok(context)
    }

    pub fn embed_ids(&self) -> &[usize] {
        &self.embed_ids
    }
    pub fn source_col_map(&self) -> HashMap<String, usize> {
        self.source_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect()
    }
}
