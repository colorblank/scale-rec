//! 推理引擎：FeatureDag + Box<dyn Model> + 预编译执行计划。
use std::collections::{HashMap, HashSet};

use candle_core::{Device, Tensor};

use crate::feats::dag::{FeatureDag, FeatureValue};
use crate::feats::ops::Fv;
use crate::models::Model;

pub struct InferenceEngine {
    pub dag: FeatureDag,
    pub model: Box<dyn Model>,
    pub embed_names: Vec<String>,
    // Pre-cached plan data
    embed_ids: Vec<usize>,
    source_col_map: HashMap<String, usize>,
    user_op_indices: HashSet<usize>,
    op_source_kind: HashMap<String, String>,
}

pub type FeatureRow = HashMap<String, serde_json::Value>;
pub type PredictionRow = HashMap<String, f32>;

impl InferenceEngine {
    pub fn new(dag: FeatureDag, model: Box<dyn Model>, embed_names: Vec<String>) -> Self {
        let embed_ids = dag.plan.embed_ids().to_vec();
        let source_col_map = dag.plan.source_col_map();
        let op_kind = dag.op_source_kind();
        let user_ops: HashSet<String> = op_kind
            .iter()
            .filter(|(_, &k)| k == "user")
            .map(|(n, _)| n.clone())
            .collect();
        let user_op_indices: HashSet<usize> = dag
            .plan
            .steps
            .iter()
            .filter(|s| user_ops.contains(&dag.execution_order[s.op_idx]))
            .map(|s| s.op_idx)
            .collect();
        let op_source_kind: HashMap<String, String> = op_kind
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        Self {
            dag,
            model,
            embed_names,
            embed_ids,
            source_col_map,
            user_op_indices,
            op_source_kind,
        }
    }

    pub fn predict(&self, features: &[FeatureRow]) -> Result<Vec<PredictionRow>, String> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        let n = features.len();

        // Build columns from rows
        let mut columns: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for row in features {
            for (key, val) in row {
                columns
                    .entry(key.clone())
                    .or_insert_with(|| Vec::with_capacity(n))
                    .push(json_to_feature(val));
            }
        }

        // Plan-based execution: zero HashMap during operator loop
        let context = self
            .dag
            .plan
            .execute_plan(&columns, &HashSet::new(), &HashMap::new())
            .map_err(|e| format!("DAG: {}", e))?;

        self.extract_predictions(&context, n)
    }

    pub fn predict_broadcast(
        &self,
        user: &FeatureRow,
        items: &[FeatureRow],
    ) -> Result<Vec<PredictionRow>, String> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // Step 1: Precompute with one item to get user-derived outputs
        let mut one: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for (k, v) in user {
            one.insert(k.clone(), vec![json_to_feature(v)]);
        }
        for (k, v) in &items[0] {
            one.insert(k.clone(), vec![json_to_feature(v)]);
        }
        let full_1 = self
            .dag
            .plan
            .execute_plan(&one, &HashSet::new(), &HashMap::new())
            .map_err(|e| format!("precompute: {}", e))?;

        // Extract user-derived outputs (column IDs from user-only ops)
        let mut precomputed: HashMap<usize, Fv> = HashMap::new();
        for step in &self.dag.plan.steps {
            if !self.user_op_indices.contains(&step.op_idx) {
                continue;
            }
            for &cid in &step.output_cols {
                if let Some(col) = full_1.get(cid) {
                    if !col.is_empty() {
                        precomputed.insert(cid, col[0].clone());
                    }
                }
            }
        }

        // Step 2: Build batch columns
        let n = items.len();
        let mut columns: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for (k, v) in user {
            columns.insert(k.clone(), vec![json_to_feature(v); n]);
        }
        for item in items {
            for (k, v) in item {
                columns
                    .entry(k.clone())
                    .or_insert_with(|| Vec::with_capacity(n))
                    .push(json_to_feature(v));
            }
        }

        let context = self
            .dag
            .plan
            .execute_plan(&columns, &self.user_op_indices, &precomputed)
            .map_err(|e| format!("DAG: {}", e))?;

        self.extract_predictions(&context, n)
    }

    fn extract_predictions(
        &self,
        context: &[Vec<Fv>],
        n: usize,
    ) -> Result<Vec<PredictionRow>, String> {
        let mut all_indices: HashMap<String, Vec<u32>> = self
            .embed_names
            .iter()
            .map(|name| (name.clone(), Vec::with_capacity(n)))
            .collect();
        for (i, name) in self.embed_names.iter().enumerate() {
            let cid = self.embed_ids[i];
            let col = context
                .get(cid)
                .ok_or_else(|| format!("Feature '{}' missing", name))?;
            all_indices.insert(
                name.clone(),
                col.iter()
                    .map(|val| match val {
                        Fv::Int(i) => *i as u32,
                        Fv::IntList(l) => l.first().copied().unwrap_or(0) as u32,
                        _ => 0,
                    })
                    .collect(),
            );
        }
        let tensor_inputs: HashMap<String, Tensor> = all_indices
            .iter()
            .map(|(n, indices)| {
                (
                    n.clone(),
                    Tensor::from_slice(indices, indices.len(), &Device::Cpu).unwrap(),
                )
            })
            .collect();
        let outputs = self
            .model
            .forward(&tensor_inputs)
            .map_err(|e| format!("Model: {}", e))?;
        let mut out_keys: Vec<&String> = outputs.keys().collect();
        out_keys.sort();
        let mut result: Vec<PredictionRow> = vec![HashMap::new(); n];
        for key in &out_keys {
            let vals: Vec<f32> = outputs
                .get(*key)
                .unwrap()
                .to_vec2::<f32>()
                .map_err(|e| format!("{}", e))?
                .iter()
                .map(|row| row[0])
                .collect();
            for (i, v) in vals.iter().enumerate() {
                result[i].insert(key.to_string(), *v);
            }
        }
        Ok(result)
    }
}

fn json_to_feature(val: &serde_json::Value) -> FeatureValue {
    match val {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Fv::Int(i as i32)
            } else {
                Fv::Float(n.as_f64().unwrap_or(0.0) as f32)
            }
        }
        serde_json::Value::String(s) => Fv::Str(s.clone()),
        serde_json::Value::Array(arr) => Fv::StrList(
            arr.iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect(),
        ),
        _ => Fv::Int(0),
    }
}
