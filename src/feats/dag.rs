//! 特征 DAG 执行器：拓扑排序、算子调度、单样本执行。
use crate::feats::config::{DType, FlowConfig, OperatorDef, SourceDef};
use crate::feats::debug::DebugTracer;
use crate::feats::ops::{
    Bucketing, CrossFeature, CustomOp, DictMapper, ExpressionOp, Fv, ListOverlap, PluginOp,
    SequenceOp, StringConcatHash, StringParser,
};
use petgraph::algo::toposort;
use petgraph::prelude::DiGraph;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub type FeatureValue = Fv;

/// 特征处理结果：包含所有特征值并区分来源。
pub struct FeatureResult {
    pub features: HashMap<String, FeatureValue>,
    pub source_names: HashSet<String>,
    pub computed_names: HashSet<String>,
}

/// 特征 DAG 执行引擎。
///
/// 根据 FlowConfig 构建算子图，拓扑排序后按序执行单样本处理。
pub struct FeatureDag {
    sources: HashMap<String, SourceDef>,
    nodes: HashMap<String, Box<dyn CustomOp>>,
    node_defs: HashMap<String, OperatorDef>,
    execution_order: Vec<String>,
    debug_mode: bool,
    pub tracer: Option<DebugTracer>,
}

impl FeatureDag {
    fn parse_default(val_str: &str, dtype: &DType) -> FeatureValue {
        match dtype {
            DType::Int => Fv::Int(val_str.parse::<i32>().unwrap_or(0)),
            DType::Float => Fv::Float(val_str.parse::<f32>().unwrap_or(0.0)),
            DType::String => Fv::Str(val_str.to_string()),
            DType::List { dtype: inner, length } => match inner.as_ref() {
                DType::Int => Fv::IntList(vec![val_str.parse::<i32>().unwrap_or(0); *length]),
                DType::Float => Fv::IntList(vec![0i32; *length]),
                DType::String => Fv::StrList(vec![val_str.to_string(); *length]),
                _ => Fv::Int(0),
            },
        }
    }

    pub fn from_config(
        config: FlowConfig,
        debug_mode: bool,
        tracer: Option<DebugTracer>,
    ) -> Result<Self, String> {
        let mut sources = HashMap::new();
        for s in config.sources {
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
        let execution_order = sorted_indices
            .iter()
            .map(|&idx| graph[idx].clone())
            .collect();
        Ok(Self {
            sources,
            nodes,
            node_defs,
            execution_order,
            debug_mode,
            tracer,
        })
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
            "StringConcatHash" => {
                let vocab_size = Self::yaml_i64(p, "vocab_size").unwrap_or(1000) as usize;
                let oov_reserve = Self::yaml_i64(p, "oov_reserve").unwrap_or(0) as usize;
                let hash_map_path = Self::yaml_str(p, "hash_map_path").unwrap_or("").to_string();
                let mode = Self::yaml_str(p, "mode").unwrap_or("train").to_string();
                let separator = Self::yaml_str(p, "separator").unwrap_or("|").to_string();
                Ok(Box::new(StringConcatHash::new(
                    vocab_size,
                    oov_reserve,
                    hash_map_path,
                    mode,
                    separator,
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
        for (name, src) in &self.sources {
            if let Some(ref emb) = src.embed {
                result.push((name.as_str(), emb));
            }
        }
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
        // Determine which source each feature name depends on
        let mut feat_kind: HashMap<String, &str> = HashMap::new();
        for (name, _) in &self.sources {
            let k = if name.starts_with("user_") || name == "user_id" { "user" }
                    else if name.starts_with("item_") || name == "item_id" { "item" }
                    else { "other" };
            feat_kind.insert(name.clone(), k);
        }
        // Propagate through operators in topological order
        for node_name in &self.execution_order {
            let def = &self.node_defs[node_name];
            let kinds: Vec<&&str> = def.inputs.iter()
                .filter_map(|inp| feat_kind.get(inp)).collect();
            let k = if kinds.iter().any(|k| **k == "user") && kinds.iter().any(|k| **k == "item") {
                "cross"
            } else if kinds.iter().any(|k| **k == "user") { "user" }
            else if kinds.iter().any(|k| **k == "item") { "item" }
            else { kinds.first().copied().unwrap_or(&&"other") };
            for out_name in &def.outputs {
                feat_kind.insert(out_name.clone(), k);
            }
        }
        // Map operator name → kind
        let mut op_kind: HashMap<String, &str> = HashMap::new();
        for node_name in &self.execution_order {
            let def = &self.node_defs[node_name];
            let kinds: Vec<&&str> = def.inputs.iter()
                .filter_map(|inp| feat_kind.get(inp)).collect();
            let k = if kinds.iter().any(|k| **k == "user") && kinds.iter().any(|k| **k == "item") {
                "cross"
            } else if kinds.iter().any(|k| **k == "user") { "user" }
            else if kinds.iter().any(|k| **k == "item") { "item" }
            else { "other" };
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
                let default_val = Self::parse_default(&source_def.default_val, &source_def.dtype);
                context.insert(name.clone(), default_val);
            }
        }

        // tracer disabled — needs update for Fv enum
        let _ = &self.tracer;

        let source_names: HashSet<String> = self.sources.keys().cloned().collect();
        let mut computed_names = HashSet::new();
        for node_name in &self.execution_order {
            let op = &self.nodes[node_name];
            let def = &self.node_defs[node_name];
            let op_inputs: Vec<Fv> = def
                .inputs
                .iter()
                .map(|inp| {
                    context.get(inp).map(|v| v.clone())
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
                let default = Self::parse_default(&source_def.default_val, &source_def.dtype);
                context.insert(name.clone(), vec![default; n_rows]);
            }
        }
        // Stage 2.5: inject precomputed values (broadcast single value to N rows)
        for (name, val) in precomputed {
            context.insert(name.clone(), vec![val.clone(); n_rows]);
        }

        // Stage 3: execute operators in topological order (bulk, zero-copy refs)
        let default_col: Vec<FeatureValue> = vec![Fv::Int(0); n_rows];
        for node_name in &self.execution_order {
            if skip_ops.contains(node_name) { continue; }
            let op = &self.nodes[node_name];
            let def = &self.node_defs[node_name];

            let input_slices: Vec<&[FeatureValue]> = def
                .inputs
                .iter()
                .map(|inp| context.get(inp).map(|c| c.as_slice()).unwrap_or(default_col.as_slice()))
                .collect();

            let result_vec = op.process_batch(&input_slices, n_rows)
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
