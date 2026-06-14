//! 特征信息视图：从 DAG 构建结果投影，提供模型构建和广播策略所需的元数据查询。
use crate::feats::config::{EmbedConfig, OperatorDef, SourceDef};
use crate::feats::schema::FeatureSchema;
use std::collections::HashMap;

/// DAG 的只读元数据视图，用于模型构建和 broadcast 策略计算。
pub struct FeatureInfo {
    sources: HashMap<String, SourceDef>,
    node_defs: HashMap<String, OperatorDef>,
    feature_schemas: HashMap<String, FeatureSchema>,
    execution_order: Vec<String>,
}

impl FeatureInfo {
    pub fn new(
        sources: HashMap<String, SourceDef>,
        node_defs: HashMap<String, OperatorDef>,
        feature_schemas: HashMap<String, FeatureSchema>,
        execution_order: Vec<String>,
    ) -> Self {
        Self {
            sources,
            node_defs,
            feature_schemas,
            execution_order,
        }
    }

    pub fn embeddable_features(&self) -> Vec<(&str, &EmbedConfig)> {
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

    pub fn op_source_kind(&self) -> HashMap<String, &str> {
        let mut feat_kind: HashMap<String, &str> = HashMap::new();
        for (name, src) in &self.sources {
            let k = match src.source.as_deref() {
                Some("User") | Some("Context") => "user",
                _ => "item",
            };
            feat_kind.insert(name.clone(), k);
        }
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

    pub fn source_defs(&self) -> &HashMap<String, SourceDef> {
        &self.sources
    }

    pub fn source_names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }

    pub fn op_outputs(&self, op_name: &str) -> Option<&Vec<String>> {
        self.node_defs.get(op_name).map(|d| &d.outputs)
    }
}
