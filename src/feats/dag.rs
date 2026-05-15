//! 特征 DAG 执行器：拓扑排序、算子调度、单样本执行。
use crate::feats::config::{DType, FlowConfig, OperatorDef, SourceDef};
use crate::feats::debug::DebugTracer;
use crate::feats::ops::{
    Bucketing, CrossFeature, CustomOp, DictMapper, ExpressionOp, ListOverlap, PluginOp, SequenceOp,
    StringConcatHash, StringParser,
};
use petgraph::algo::toposort;
use petgraph::prelude::DiGraph;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub type FeatureValue = Arc<dyn Any + Send + Sync>;

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
            DType::Int => Arc::new(val_str.parse::<i32>().unwrap_or(0)),
            DType::Float => Arc::new(val_str.parse::<f32>().unwrap_or(0.0)),
            DType::String => Arc::new(val_str.to_string()),
            DType::List {
                dtype: inner,
                length,
            } => match inner.as_ref() {
                DType::Int => Arc::new(vec![val_str.parse::<i32>().unwrap_or(0); *length]),
                DType::Float => Arc::new(vec![val_str.parse::<f32>().unwrap_or(0.0); *length]),
                DType::String => Arc::new(vec![val_str.to_string(); *length]),
                _ => Arc::new(0i32),
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
                context.insert(name.clone(), Arc::clone(val));
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

        if let Some(ref tracer) = self.tracer {
            let snapshot: HashMap<String, &(dyn Any + Send + Sync)> = context
                .iter()
                .map(|(k, v)| (k.clone(), v.as_ref() as &(dyn Any + Send + Sync)))
                .collect();
            tracer.trace_defaults(&snapshot);
            tracer.trace_overrides(&snapshot, overridden);
        }

        let source_names: HashSet<String> = self.sources.keys().cloned().collect();
        let mut computed_names = HashSet::new();
        for node_name in &self.execution_order {
            let op = &self.nodes[node_name];
            let def = &self.node_defs[node_name];
            let mut op_inputs: Vec<&(dyn Any + Send + Sync)> = Vec::new();
            for input_name in &def.inputs {
                let val = context
                    .get(input_name)
                    .ok_or_else(|| format!("Required input '{}' not found", input_name))?;
                op_inputs.push(val.as_ref());
            }
            let output = op.process(&op_inputs)?;

            // Trace operator I/O
            if let Some(ref tracer) = self.tracer {
                tracer.trace_operator(
                    node_name,
                    &def.inputs,
                    &op_inputs,
                    &def.outputs,
                    output.as_ref(),
                );
            }

            let output_arc: FeatureValue = Arc::from(output);
            for out_name in &def.outputs {
                context.insert(out_name.clone(), Arc::clone(&output_arc));
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

    pub fn dump_snapshot(
        &self,
        context: &HashMap<String, FeatureValue>,
        source_names: &HashSet<String>,
        computed_names: &HashSet<String>,
    ) {
        println!("[Feature Snapshot]");
        for (name, val) in context {
            let inner = val.as_ref();
            let type_name = if inner.downcast_ref::<i32>().is_some() {
                "i32"
            } else if inner.downcast_ref::<f32>().is_some() {
                "f32"
            } else if inner.downcast_ref::<String>().is_some() {
                "String"
            } else if inner.downcast_ref::<Vec<String>>().is_some() {
                "Vec<String>"
            } else if inner.downcast_ref::<Vec<i32>>().is_some() {
                "Vec<i32>"
            } else {
                "unknown"
            };
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
