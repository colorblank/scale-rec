//! 推理引擎：FeatureDag + Box<dyn Model> 封装。
use std::collections::{HashMap, HashSet};

use candle_core::{Device, Tensor};

use crate::feats::dag::{FeatureDag, FeatureValue};
use crate::feats::ops::Fv;
use crate::models::Model;

/// 推理引擎：持有 DAG 和已加载权重的模型。
pub struct InferenceEngine {
    pub dag: FeatureDag,
    pub model: Box<dyn Model>,
    pub embed_names: Vec<String>,
}

/// 预测请求中的单行特征（JSON 反序列化）。
pub type FeatureRow = HashMap<String, serde_json::Value>;

/// 预测结果中的单行输出。
pub type PredictionRow = HashMap<String, f32>;

impl InferenceEngine {
    /// 执行 point-wise 推理：每行是一个完整样本。
    pub fn predict(&self, features: &[FeatureRow]) -> Result<Vec<PredictionRow>, String> {
        if features.is_empty() {
            return Ok(vec![]);
        }

        let n = features.len();

        // Build columnar input from rows
        let mut columns: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for row in features {
            for (key, val) in row {
                columns.entry(key.clone())
                    .or_insert_with(|| Vec::with_capacity(n))
                    .push(json_to_feature(val));
            }
        }

        // Batch DAG execution — one pass through all operators on full columns
        let batch_result = self.dag.execute_batch(&columns, &Default::default())
            .map_err(|e| format!("DAG error: {}", e))?;

        // Extract embeddable feature indices from batch result
        let mut all_indices: HashMap<String, Vec<u32>> = self
            .embed_names
            .iter()
            .map(|n| (n.clone(), Vec::with_capacity(features.len())))
            .collect();
        for name in &self.embed_names {
            let col = batch_result.get(name)
                .ok_or_else(|| format!("Feature '{}' not found in batch output", name))?;
            let indices: Vec<u32> = col.iter().map(|val| {
                match val.clone() {
                    Fv::Int(i) => i as u32,
                    Fv::IntList(ref list) => list.first().copied().unwrap_or(0) as u32,
                    _ => 0,
                }
            }).collect();
            all_indices.insert(name.clone(), indices);
        }

        let tensor_inputs: HashMap<String, Tensor> = all_indices
            .iter()
            .map(|(name, indices)| {
                let t = Tensor::from_slice(indices, indices.len(), &Device::Cpu).unwrap();
                (name.clone(), t)
            })
            .collect();

        let outputs = self
            .model
            .forward(&tensor_inputs)
            .map_err(|e| format!("Model error: {}", e))?;

        let mut out_keys: Vec<&String> = outputs.keys().collect();
        out_keys.sort();
        let mut result: Vec<PredictionRow> = vec![HashMap::new(); n];
        for key in &out_keys {
            let tensor = outputs.get(*key).unwrap();
            let vals: Vec<f32> = tensor
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

    /// 广播模式：预计算用户特征一次，广播到 N 个物品行，跳过用户专属算子。
    pub fn predict_broadcast(
        &self,
        user: &FeatureRow,
        items: &[FeatureRow],
    ) -> Result<Vec<PredictionRow>, String> {
        if items.is_empty() { return Ok(vec![]); }

        // Step 1: Precompute user outputs with first item (1-row batch)
        let mut pre_cols: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for (k, v) in user { pre_cols.insert(k.clone(), vec![json_to_feature(v)]); }
        for (k, v) in &items[0] { pre_cols.insert(k.clone(), vec![json_to_feature(v)]); }
        let full_1row = self.dag.execute_batch(&pre_cols, &Default::default())
            .map_err(|e| format!("precompute: {}", e))?;

        // Step 2: Classify operators, extract user-only outputs
        let op_kind = self.dag.op_source_kind();
        let mut user_ops: HashSet<String> = HashSet::new();
        let mut precomputed: HashMap<String, FeatureValue> = HashMap::new();
        for (op_name, &kind) in &op_kind {
            if kind != "user" { continue; }
            user_ops.insert(op_name.clone());
            if let Some(outputs) = self.dag.op_outputs(op_name) {
                for out_name in outputs {
                    if let Some(col) = full_1row.get(out_name) {
                        if !col.is_empty() { precomputed.insert(out_name.clone(), col[0].clone()); }
                    }
                }
            }
        }

        // Step 3: Build N-row batch columns — user sources broadcast, item sources per row
        let n = items.len();
        let mut columns: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        // User source features: broadcast to N rows
        for (k, v) in user {
            columns.insert(k.clone(), vec![json_to_feature(v); n]);
        }
        // Item source features: one per row
        for item in items {
            for (k, v) in item {
                columns.entry(k.clone()).or_insert_with(|| Vec::with_capacity(n))
                    .push(json_to_feature(v));
            }
        }

        // Step 4: Execute batch with precomputed user outputs, skipping user ops
        let batch_result = self.dag.execute_batch_precomputed(
            &columns, &user_ops, &precomputed,
        ).map_err(|e| format!("DAG: {}", e))?;

        // Step 5: Extract predictions (same as predict())
        let mut all_indices: HashMap<String, Vec<u32>> = self.embed_names.iter()
            .map(|n| (n.clone(), Vec::with_capacity(items.len()))).collect();
        for name in &self.embed_names {
            let col = batch_result.get(name)
                .ok_or_else(|| format!("Feature '{}' missing", name))?;
            all_indices.insert(name.clone(), col.iter().map(|val| match val.clone() {
                Fv::Int(i) => i as u32,
                Fv::IntList(ref l) => l.first().copied().unwrap_or(0) as u32,
                _ => 0,
            }).collect());
        }
        let tensor_inputs: HashMap<String, Tensor> = all_indices.iter()
            .map(|(n, indices)| (n.clone(),
                Tensor::from_slice(indices, indices.len(), &Device::Cpu).unwrap())).collect();
        let outputs = self.model.forward(&tensor_inputs)
            .map_err(|e| format!("Model: {}", e))?;
        let mut out_keys: Vec<&String> = outputs.keys().collect(); out_keys.sort();
        let mut result: Vec<PredictionRow> = vec![HashMap::new(); n];
        for key in &out_keys {
            let vals: Vec<f32> = outputs.get(*key).unwrap()
                .to_vec2::<f32>().map_err(|e| format!("{}", e))?
                .iter().map(|row| row[0]).collect();
            for (i, v) in vals.iter().enumerate() { result[i].insert(key.to_string(), *v); }
        }
        Ok(result)
    }
}

fn json_to_feature(val: &serde_json::Value) -> FeatureValue {
    match val {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() { Fv::Int(i as i32) }
            else { Fv::Float(n.as_f64().unwrap_or(0.0) as f32) }
        }
        serde_json::Value::String(s) => Fv::Str(s.clone()),
        serde_json::Value::Array(arr) => {
            Fv::StrList(arr.iter().map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            }).collect())
        }
        _ => Fv::Int(0),
    }
}
