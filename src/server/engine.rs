//! 推理引擎：FeatureDag + Box<dyn Model> + 预编译执行计划。
use std::collections::{HashMap, HashSet};

use candle_core::{Device, Tensor};

use crate::feats::config::{DType, PoolingStrategy, SourceDef};
use crate::feats::dag::{FeatureDag, FeatureValue};
use crate::feats::ops::Fv;
use crate::layers::embedding::FeatureSpec;
use crate::models::Model;

pub struct InferenceEngine {
    pub dag: FeatureDag,
    pub model: Box<dyn Model>,
    pub embed_features: Vec<FeatureSpec>,
    // Pre-cached plan data
    embed_ids: Vec<usize>,
    user_op_indices: HashSet<usize>,
}

pub type FeatureRow = HashMap<String, serde_json::Value>;
pub type PredictionRow = HashMap<String, f32>;

impl InferenceEngine {
    pub fn new(dag: FeatureDag, model: Box<dyn Model>, embed_features: Vec<FeatureSpec>) -> Self {
        let embed_ids = dag.plan.embed_ids().to_vec();
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
        Self {
            dag,
            model,
            embed_features,
            embed_ids,
            user_op_indices,
        }
    }

    pub fn predict(&self, features: &[FeatureRow]) -> Result<Vec<PredictionRow>, String> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        let n = features.len();

        let columns = self.rows_to_columns(features);

        // Plan-based execution: zero HashMap during operator loop
        let context = self
            .dag
            .plan
            .execute_plan(&columns, &HashSet::new(), &HashMap::new())
            .map_err(|e| format!("DAG: {}", e))?;

        self.extract_predictions(&context, n)
    }

    fn rows_to_columns(&self, rows: &[FeatureRow]) -> HashMap<String, Vec<FeatureValue>> {
        let n = rows.len();
        let mut columns: HashMap<String, Vec<FeatureValue>> = self
            .dag
            .source_defs()
            .iter()
            .map(|(name, source)| (name.clone(), vec![source_default(source); n]))
            .collect();

        for (row_idx, row) in rows.iter().enumerate() {
            for (key, val) in row {
                if let Some(col) = columns.get_mut(key) {
                    col[row_idx] = json_to_feature(val);
                }
            }
        }
        columns
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
        let merged_rows: Vec<FeatureRow> = items
            .iter()
            .map(|item| {
                let mut row = user.clone();
                row.extend(item.clone());
                row
            })
            .collect();
        let columns = self.rows_to_columns(&merged_rows);

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
        let mut tensor_inputs: HashMap<String, Tensor> =
            HashMap::with_capacity(self.embed_features.len());
        for (i, spec) in self.embed_features.iter().enumerate() {
            let cid = self.embed_ids[i];
            let col = context
                .get(cid)
                .ok_or_else(|| format!("Feature '{}' missing", spec.name))?;
            let tensor = feature_column_to_tensor(spec, col, n)?;
            tensor_inputs.insert(spec.name.clone(), tensor);
        }
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

fn feature_column_to_tensor(spec: &FeatureSpec, col: &[Fv], n: usize) -> Result<Tensor, String> {
    let use_sequence =
        spec.pooling != PoolingStrategy::First && col.iter().any(|v| matches!(v, Fv::IntList(_)));

    if use_sequence {
        let observed_max = col
            .iter()
            .filter_map(|v| match v {
                Fv::IntList(values) => Some(values.len()),
                _ => None,
            })
            .max()
            .unwrap_or(1);
        let seq_len = spec.seq_len.unwrap_or(observed_max).max(1);
        let mut flat = Vec::with_capacity(n * seq_len);
        for val in col.iter().take(n) {
            match val {
                Fv::IntList(values) => {
                    for idx in 0..seq_len {
                        flat.push(values.get(idx).copied().unwrap_or(0).max(0) as u32);
                    }
                }
                Fv::Int(i) => {
                    flat.push((*i).max(0) as u32);
                    flat.extend(std::iter::repeat(0).take(seq_len - 1));
                }
                _ => flat.extend(std::iter::repeat(0).take(seq_len)),
            }
        }
        return Tensor::from_slice(flat.as_slice(), (n, seq_len), &Device::Cpu)
            .map_err(|e| format!("tensor '{}': {}", spec.name, e));
    }

    let indices: Vec<u32> = col
        .iter()
        .take(n)
        .map(|val| match val {
            Fv::Int(i) => (*i).max(0) as u32,
            Fv::IntList(values) => values.first().copied().unwrap_or(0).max(0) as u32,
            _ => 0,
        })
        .collect();
    Tensor::from_slice(indices.as_slice(), indices.len(), &Device::Cpu)
        .map_err(|e| format!("tensor '{}': {}", spec.name, e))
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

fn source_default(source: &SourceDef) -> FeatureValue {
    match &source.dtype {
        DType::Int => Fv::Int(source.default_val.parse::<i32>().unwrap_or(0)),
        DType::Float => Fv::Float(source.default_val.parse::<f32>().unwrap_or(0.0)),
        DType::String => Fv::Str(source.default_val.clone()),
        DType::List { dtype, length } => match dtype.as_ref() {
            DType::Int => Fv::IntList(vec![
                source.default_val.parse::<i32>().unwrap_or(0);
                *length
            ]),
            DType::Float => Fv::FloatList(vec![
                source.default_val.parse::<f32>().unwrap_or(0.0);
                *length
            ]),
            DType::String => Fv::StrList(vec![source.default_val.clone(); *length]),
            _ => Fv::Int(0),
        },
    }
}
