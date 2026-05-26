//! 推理引擎：FeatureDag + Box<dyn Model> + 预编译执行计划。
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use candle_core::{Device, Tensor};

use crate::feats::config::{DType, PoolingStrategy, SourceDef};
use crate::feats::dag::{FeatureDag, FeatureValue};
use crate::feats::ops::Fv;
use crate::layers::embedding::FeatureSpec;
use crate::models::Model;

#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub parse_us: u64,
    pub dag_us: u64,
    pub tensor_us: u64,
    pub forward_us: u64,
    pub response_us: u64,
}

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

    pub fn predict(
        &self,
        features: &[FeatureRow],
    ) -> Result<(Vec<PredictionRow>, InferenceMetrics), String> {
        let mut metrics = InferenceMetrics::default();
        if features.is_empty() {
            return Ok((vec![], metrics));
        }
        let n = features.len();

        let start_parse = Instant::now();
        let columns = self.rows_to_columns(features)?;
        metrics.parse_us = start_parse.elapsed().as_micros() as u64;

        // Plan-based execution: zero HashMap during operator loop
        let start_dag = Instant::now();
        let context = self
            .dag
            .plan
            .execute_plan(&columns, &HashSet::new(), &HashMap::new())
            .map_err(|e| format!("DAG: {}", e))?;
        metrics.dag_us = start_dag.elapsed().as_micros() as u64;

        let (predictions, tensor_us, forward_us, response_us) =
            self.extract_predictions_measured(&context, n)?;
        metrics.tensor_us = tensor_us;
        metrics.forward_us = forward_us;
        metrics.response_us = response_us;

        Ok((predictions, metrics))
    }

    fn rows_to_columns(
        &self,
        rows: &[FeatureRow],
    ) -> Result<HashMap<String, Vec<FeatureValue>>, String> {
        let n = rows.len();
        let mut columns: HashMap<String, Vec<FeatureValue>> = self
            .dag
            .source_defs()
            .iter()
            .map(|(name, source)| (name.clone(), vec![source_default(source); n]))
            .collect();

        let mut default_hits = 0;
        let mut empty_sequences = 0;

        for (row_idx, row) in rows.iter().enumerate() {
            for (key, _source) in self.dag.source_defs() {
                if !row.contains_key(key) {
                    default_hits += 1;
                    tracing::debug!(key = %key, row = row_idx, "default value hit");
                }
            }

            for (key, val) in row {
                if let Some(col) = columns.get_mut(key) {
                    let source = self.dag.source_defs().get(key).unwrap();
                    let fv = json_to_feature_typed(val, &source.dtype)
                        .map_err(|err| format!("column '{}' row {}: {}", key, row_idx, err))?;

                    match &fv {
                        Fv::IntList(v) if v.is_empty() => {
                            empty_sequences += 1;
                            tracing::warn!(key = %key, row = row_idx, "empty sequence detected");
                        }
                        Fv::FloatList(v) if v.is_empty() => {
                            empty_sequences += 1;
                            tracing::warn!(key = %key, row = row_idx, "empty sequence detected");
                        }
                        Fv::StrList(v) if v.is_empty() => {
                            empty_sequences += 1;
                            tracing::warn!(key = %key, row = row_idx, "empty sequence detected");
                        }
                        _ => {}
                    }

                    col[row_idx] = fv;
                }
            }
        }

        if default_hits > 0 {
            tracing::info!(
                count = default_hits,
                "default values used during rows_to_columns"
            );
        }
        if empty_sequences > 0 {
            tracing::warn!(
                count = empty_sequences,
                "empty sequences detected during rows_to_columns"
            );
        }

        Ok(columns)
    }

    pub fn predict_broadcast(
        &self,
        user: &FeatureRow,
        items: &[FeatureRow],
    ) -> Result<(Vec<PredictionRow>, InferenceMetrics), String> {
        let mut metrics = InferenceMetrics::default();
        if items.is_empty() {
            return Ok((vec![], metrics));
        }

        // Step 1: Precompute with one item to get user-derived outputs
        let start_parse1 = Instant::now();
        let mut one: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for (k, v) in user {
            if let Some(source) = self.dag.source_defs().get(k) {
                let fv = json_to_feature_typed(v, &source.dtype)
                    .map_err(|err| format!("user field '{}': {}", k, err))?;
                one.insert(k.clone(), vec![fv]);
            }
        }
        for (k, v) in &items[0] {
            if let Some(source) = self.dag.source_defs().get(k) {
                let fv = json_to_feature_typed(v, &source.dtype)
                    .map_err(|err| format!("item[0] field '{}': {}", k, err))?;
                one.insert(k.clone(), vec![fv]);
            }
        }
        metrics.parse_us = start_parse1.elapsed().as_micros() as u64;

        let start_dag1 = Instant::now();
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
        let dag1_us = start_dag1.elapsed().as_micros() as u64;

        // Step 2: Build batch columns
        let start_parse2 = Instant::now();
        let n = items.len();
        let merged_rows: Vec<FeatureRow> = items
            .iter()
            .map(|item| {
                let mut row = user.clone();
                row.extend(item.clone());
                row
            })
            .collect();
        let columns = self.rows_to_columns(&merged_rows)?;
        metrics.parse_us += start_parse2.elapsed().as_micros() as u64;

        let start_dag2 = Instant::now();
        let context = self
            .dag
            .plan
            .execute_plan(&columns, &self.user_op_indices, &precomputed)
            .map_err(|e| format!("DAG: {}", e))?;
        metrics.dag_us = dag1_us + start_dag2.elapsed().as_micros() as u64;

        let (predictions, tensor_us, forward_us, response_us) =
            self.extract_predictions_measured(&context, n)?;
        metrics.tensor_us = tensor_us;
        metrics.forward_us = forward_us;
        metrics.response_us = response_us;

        Ok((predictions, metrics))
    }

    fn extract_predictions_measured(
        &self,
        context: &[Vec<Fv>],
        n: usize,
    ) -> Result<(Vec<PredictionRow>, u64, u64, u64), String> {
        let start_tensor = Instant::now();
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
        let tensor_us = start_tensor.elapsed().as_micros() as u64;

        let start_forward = Instant::now();
        let outputs = self
            .model
            .forward(&tensor_inputs)
            .map_err(|e| format!("Model: {}", e))?;
        let forward_us = start_forward.elapsed().as_micros() as u64;

        let start_response = Instant::now();
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
        let response_us = start_response.elapsed().as_micros() as u64;

        Ok((result, tensor_us, forward_us, response_us))
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

fn json_to_feature_typed(val: &serde_json::Value, dtype: &DType) -> Result<FeatureValue, String> {
    match val {
        serde_json::Value::Null => Err("explicit null value is not allowed".to_string()),
        serde_json::Value::Bool(b) => match dtype {
            DType::Int => Ok(Fv::Int(if *b { 1 } else { 0 })),
            DType::String => Ok(Fv::Str(b.to_string())),
            _ => Err(format!("cannot convert bool to dtype {:?}", dtype)),
        },
        serde_json::Value::Number(n) => match dtype {
            DType::Int => {
                if let Some(i) = n.as_i64() {
                    i32::try_from(i)
                        .map(Fv::Int)
                        .map_err(|_| format!("integer value out of range for i32: {}", n))
                } else if let Some(u) = n.as_u64() {
                    i32::try_from(u)
                        .map(Fv::Int)
                        .map_err(|_| format!("integer value out of range for i32: {}", n))
                } else {
                    Err(format!("invalid integer value: {}", n))
                }
            }
            DType::Float => Ok(Fv::Float(
                n.as_f64()
                    .ok_or_else(|| format!("invalid float value: {}", n))? as f32,
            )),
            DType::String => Ok(Fv::Str(n.to_string())),
            _ => Err(format!("cannot convert scalar number to dtype {:?}", dtype)),
        },
        serde_json::Value::String(s) => match dtype {
            DType::Int => s
                .parse::<i32>()
                .map(Fv::Int)
                .map_err(|e| format!("parse int from '{}' failed: {}", s, e)),
            DType::Float => s
                .parse::<f32>()
                .map(Fv::Float)
                .map_err(|e| format!("parse float from '{}' failed: {}", s, e)),
            DType::String => Ok(Fv::Str(s.clone())),
            _ => Err(format!("cannot convert string to dtype {:?}", dtype)),
        },
        serde_json::Value::Array(arr) => match dtype {
            DType::List {
                dtype: elem_type, ..
            } => {
                let mut elements = Vec::with_capacity(arr.len());
                for (idx, item) in arr.iter().enumerate() {
                    let parsed = json_to_feature_typed(item, elem_type)
                        .map_err(|e| format!("at index {}: {}", idx, e))?;
                    elements.push(parsed);
                }
                match elem_type.as_ref() {
                    DType::Int => {
                        let mut ints = Vec::with_capacity(elements.len());
                        for el in elements {
                            if let Fv::Int(i) = el {
                                ints.push(i);
                            } else {
                                return Err("expected Int element".to_string());
                            }
                        }
                        Ok(Fv::IntList(ints))
                    }
                    DType::Float => {
                        let mut floats = Vec::with_capacity(elements.len());
                        for el in elements {
                            if let Fv::Float(f) = el {
                                floats.push(f);
                            } else {
                                return Err("expected Float element".to_string());
                            }
                        }
                        Ok(Fv::FloatList(floats))
                    }
                    DType::String => {
                        let mut strs = Vec::with_capacity(elements.len());
                        for el in elements {
                            if let Fv::Str(s) = el {
                                strs.push(s);
                            } else {
                                return Err("expected String element".to_string());
                            }
                        }
                        Ok(Fv::StrList(strs))
                    }
                    _ => Err("unsupported list element dtype".to_string()),
                }
            }
            _ => Err(format!("cannot convert array to scalar dtype {:?}", dtype)),
        },
        serde_json::Value::Object(_) => {
            Err("cannot convert JSON object to feature value".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_int_rejects_fractional_number() {
        let value = serde_json::json!(12.9);

        let err = json_to_feature_typed(&value, &DType::Int).unwrap_err();

        assert!(err.contains("invalid integer value"));
    }

    #[test]
    fn json_int_rejects_out_of_range_number() {
        let value = serde_json::json!(i64::from(i32::MAX) + 1);

        let err = json_to_feature_typed(&value, &DType::Int).unwrap_err();

        assert!(err.contains("out of range"));
    }

    #[test]
    fn json_int_accepts_integral_number() {
        let value = serde_json::json!(12);

        let parsed = json_to_feature_typed(&value, &DType::Int).unwrap();

        assert!(matches!(parsed, Fv::Int(12)));
    }

    #[test]
    fn json_int_list_rejects_fractional_elements() {
        let value = serde_json::json!([1, 2.5, 3]);
        let dtype = DType::List {
            dtype: Box::new(DType::Int),
            length: 3,
        };

        let err = json_to_feature_typed(&value, &dtype).unwrap_err();

        assert!(err.contains("at index 1"));
        assert!(err.contains("invalid integer value"));
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
