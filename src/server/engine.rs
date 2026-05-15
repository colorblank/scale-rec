//! 推理引擎：FeatureDag + Box<dyn Model> 封装。
use std::collections::HashMap;

use candle_core::{Device, Tensor};

use crate::feats::dag::{FeatureDag, FeatureValue};
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
        let batch_result = self.dag.execute_batch(&columns)
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
                if let Some(i) = (**val).downcast_ref::<i32>() { *i as u32 }
                else if let Some(list) = (**val).downcast_ref::<Vec<i32>>() {
                    list.first().copied().unwrap_or(0) as u32
                } else { 0 }
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

    /// 广播模式：一个 user + N 个 items → N 个预测。
    pub fn predict_broadcast(
        &self,
        user: &FeatureRow,
        items: &[FeatureRow],
    ) -> Result<Vec<PredictionRow>, String> {
        let mut rows: Vec<FeatureRow> = Vec::with_capacity(items.len());
        for item in items {
            let mut row = user.clone();
            for (k, v) in item {
                row.insert(k.clone(), v.clone());
            }
            rows.push(row);
        }
        self.predict(&rows)
    }
}

fn json_to_feature(val: &serde_json::Value) -> FeatureValue {
    use std::sync::Arc;
    match val {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Arc::new(i as i32)
            } else if let Some(f) = n.as_f64() {
                Arc::new(f as f32)
            } else {
                Arc::new(0i32)
            }
        }
        serde_json::Value::String(s) => Arc::new(s.clone()),
        serde_json::Value::Array(arr) => {
            let strs: Vec<String> = arr
                .iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect();
            Arc::new(strs)
        }
        _ => Arc::new(0i32),
    }
}
