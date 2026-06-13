//! 推理引擎：`FeatureDag` + `Box<dyn Model>` + 预编译执行计划。
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use candle_core::{Device, Tensor};

use crate::feats::config::{DType, PoolingStrategy, TruncationSide};
use crate::feats::dag::{FeatureDag, FeatureValue};
use crate::feats::defaults::source_default;
use crate::feats::ops::Fv;
use crate::layers::embedding::FeatureSpec;
use crate::models::Model;

/// 推理各阶段耗时指标（微秒）。
#[derive(Debug, Clone, Default)]
pub struct InferenceMetrics {
    pub parse_us: u64,
    pub dag_us: u64,
    pub tensor_us: u64,
    pub forward_us: u64,
    pub response_us: u64,
}

/// 推理错误类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceErrorKind {
    BadRequest,
    Feature,
    Model,
    Internal,
}

/// 推理错误：携带类型和消息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceError {
    kind: InferenceErrorKind,
    message: String,
}

impl InferenceError {
    /// 创建请求参数错误（BadRequest）。
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::BadRequest,
            message: message.into(),
        }
    }

    /// 创建特征处理错误。
    pub fn feature(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::Feature,
            message: message.into(),
        }
    }

    /// 创建模型推理错误。
    pub fn model(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::Model,
            message: message.into(),
        }
    }

    /// 创建内部错误。
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: InferenceErrorKind::Internal,
            message: message.into(),
        }
    }

    /// 返回错误类型。
    pub fn kind(&self) -> InferenceErrorKind {
        self.kind
    }

    /// 返回错误消息。
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for InferenceError {}

/// 推理结果类型别名。
pub type InferenceResult<T> = Result<T, InferenceError>;

pub struct InferenceEngine {
    pub dag: FeatureDag,
    pub model: Box<dyn Model>,
    pub embed_features: Vec<FeatureSpec>,
    pub device: Device,
    // Pre-cached plan data
    embed_ids: Vec<usize>,
    user_op_indices: HashSet<usize>,
    broadcast_precompute_skip_indices: HashSet<usize>,
}

/// 单行特征数据的类型别名。
pub type FeatureRow = HashMap<String, serde_json::Value>;
/// 单行预测结果的类型别名。
pub type PredictionRow = HashMap<String, f32>;

impl InferenceEngine {
    /// 构造推理引擎。
    pub fn new(
        dag: FeatureDag,
        model: Box<dyn Model>,
        embed_features: Vec<FeatureSpec>,
        device: Device,
    ) -> Self {
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
        let broadcast_precompute_skip_indices: HashSet<usize> = dag
            .plan
            .steps
            .iter()
            .filter(|s| !user_ops.contains(&dag.execution_order[s.op_idx]))
            .map(|s| s.op_idx)
            .collect();
        Self {
            dag,
            model,
            embed_features,
            device,
            embed_ids,
            user_op_indices,
            broadcast_precompute_skip_indices,
        }
    }

    /// 批量预测：特征行列表 → 预测结果行列表。
    pub fn predict(
        &self,
        features: &[FeatureRow],
    ) -> InferenceResult<(Vec<PredictionRow>, InferenceMetrics)> {
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
            .map_err(|e| InferenceError::feature(format!("DAG: {}", e)))?;
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
    ) -> InferenceResult<HashMap<String, Vec<FeatureValue>>> {
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
                    let source = self.dag.source_defs().get(key).ok_or_else(|| {
                        InferenceError::internal(format!("source '{}' missing from DAG", key))
                    })?;
                    let fv = json_to_feature_typed(val, &source.dtype).map_err(|err| {
                        InferenceError::bad_request(format!(
                            "column '{}' row {}: {}",
                            key, row_idx, err
                        ))
                    })?;

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

    /// 广播模式预测：用户特征一次计算 + 候选物品逐条预测。
    pub fn predict_broadcast(
        &self,
        user: &FeatureRow,
        items: &[FeatureRow],
    ) -> InferenceResult<(Vec<PredictionRow>, InferenceMetrics)> {
        let mut metrics = InferenceMetrics::default();
        if items.is_empty() {
            return Ok((vec![], metrics));
        }

        // Step 1: Precompute only the user-side subgraph once.
        // Item-side ops are skipped here and are recomputed per candidate later.
        let start_parse1 = Instant::now();
        let mut one: HashMap<String, Vec<FeatureValue>> = HashMap::new();
        for (k, v) in user {
            if let Some(source) = self.dag.source_defs().get(k) {
                let fv = json_to_feature_typed(v, &source.dtype).map_err(|err| {
                    InferenceError::bad_request(format!("user field '{}': {}", k, err))
                })?;
                one.insert(k.clone(), vec![fv]);
            }
        }
        metrics.parse_us = start_parse1.elapsed().as_micros() as u64;

        let start_dag1 = Instant::now();
        let full_1 = self
            .dag
            .plan
            .execute_plan(
                &one,
                &self.broadcast_precompute_skip_indices,
                &HashMap::new(),
            )
            .map_err(|e| InferenceError::feature(format!("precompute: {}", e)))?;

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

        // Step 2: Build batch columns without redundant cloning and parsing of user features
        let start_parse2 = Instant::now();
        let n = items.len();
        let mut columns: HashMap<String, Vec<FeatureValue>> = self
            .dag
            .source_defs()
            .iter()
            .map(|(name, source)| (name.clone(), vec![source_default(source); n]))
            .collect();

        // 1. Broadcast pre-parsed user-side features to all n rows
        for (k, val_vec) in &one {
            if let Some(col) = columns.get_mut(k) {
                if let Some(fv) = val_vec.first() {
                    for row_idx in 0..n {
                        col[row_idx] = fv.clone();
                    }
                }
            }
        }

        // 2. Parse candidate item-side features for each row, overwriting defaults or broadcasted values
        for (row_idx, item) in items.iter().enumerate() {
            for (key, val) in item {
                if let Some(col) = columns.get_mut(key) {
                    let source = self.dag.source_defs().get(key).ok_or_else(|| {
                        InferenceError::internal(format!("source '{}' missing from DAG", key))
                    })?;
                    let fv = json_to_feature_typed(val, &source.dtype).map_err(|err| {
                        InferenceError::bad_request(format!(
                            "item column '{}' row {}: {}",
                            key, row_idx, err
                        ))
                    })?;
                    col[row_idx] = fv;
                }
            }
        }
        metrics.parse_us += start_parse2.elapsed().as_micros() as u64;

        let start_dag2 = Instant::now();
        let context = self
            .dag
            .plan
            .execute_plan(&columns, &self.user_op_indices, &precomputed)
            .map_err(|e| InferenceError::feature(format!("DAG: {}", e)))?;
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
    ) -> InferenceResult<(Vec<PredictionRow>, u64, u64, u64)> {
        let start_tensor = Instant::now();
        let mut tensor_inputs: HashMap<String, Tensor> =
            HashMap::with_capacity(self.embed_features.len());
        for (i, spec) in self.embed_features.iter().enumerate() {
            let cid = self.embed_ids[i];
            let col = context.get(cid).ok_or_else(|| {
                InferenceError::feature(format!("feature '{}' missing", spec.name))
            })?;
            let tensor = feature_column_to_tensor(spec, col, n, &self.device)
                .map_err(InferenceError::model)?;
            tensor_inputs.insert(spec.name.clone(), tensor);
        }
        let tensor_us = start_tensor.elapsed().as_micros() as u64;

        let start_forward = Instant::now();
        let outputs = self
            .model
            .forward(&tensor_inputs)
            .map_err(|e| InferenceError::model(format!("model forward: {}", e)))?;
        let forward_us = start_forward.elapsed().as_micros() as u64;

        let start_response = Instant::now();
        let mut out_keys: Vec<&String> = outputs.keys().collect();
        out_keys.sort();
        let mut result: Vec<PredictionRow> = vec![HashMap::new(); n];
        for key in &out_keys {
            let vals: Vec<f32> = outputs
                .get(*key)
                .ok_or_else(|| InferenceError::model(format!("output '{}' missing", key)))?
                .flatten_all()
                .map_err(|e| InferenceError::model(format!("flatten output '{}': {}", key, e)))?
                .to_vec1::<f32>()
                .map_err(|e| InferenceError::model(format!("copy output '{}': {}", key, e)))?;
            for (i, v) in vals.iter().enumerate() {
                result[i].insert(key.to_string(), *v);
            }
        }
        let response_us = start_response.elapsed().as_micros() as u64;

        Ok((result, tensor_us, forward_us, response_us))
    }
}

fn feature_column_to_tensor(
    spec: &FeatureSpec,
    col: &[Fv],
    n: usize,
    device: &Device,
) -> Result<Tensor, String> {
    let use_sequence =
        spec.pooling != PoolingStrategy::First && col.iter().any(|v| matches!(v, Fv::IntList(_)));

    if use_sequence {
        let seq_len = spec
            .seq_len
            .ok_or_else(|| {
                format!(
                    "feature '{}' sequence pooling requires seq_len > 0",
                    spec.name
                )
            })?
            .max(1);
        let mut flat = Vec::with_capacity(n * seq_len);
        for val in col.iter().take(n) {
            match val {
                Fv::IntList(values) => {
                    let start_offset =
                        if spec.truncation == TruncationSide::Tail && values.len() > seq_len {
                            values.len() - seq_len
                        } else {
                            0
                        };
                    for idx in 0..seq_len {
                        flat.push(
                            values.get(start_offset + idx).copied().unwrap_or(0).max(0) as u32
                        );
                    }
                }
                Fv::Int(i) => {
                    flat.push((*i).max(0) as u32);
                    flat.extend(std::iter::repeat(0).take(seq_len - 1));
                }
                _ => flat.extend(std::iter::repeat(0).take(seq_len)),
            }
        }
        let cpu_tensor = Tensor::from_slice(flat.as_slice(), (n, seq_len), &Device::Cpu)
            .map_err(|e| format!("tensor '{}': {}", spec.name, e))?;
        return cpu_tensor
            .to_device(device)
            .map_err(|e| format!("tensor '{}' to device: {}", spec.name, e));
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
    let cpu_tensor = Tensor::from_slice(indices.as_slice(), indices.len(), &Device::Cpu)
        .map_err(|e| format!("tensor '{}': {}", spec.name, e))?;
    cpu_tensor
        .to_device(device)
        .map_err(|e| format!("tensor '{}' to device: {}", spec.name, e))
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
            DType::Enum { values, oov, .. } => normalize_enum_value(&n.to_string(), values, oov),
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
            DType::Enum { values, oov, .. } => normalize_enum_value(s, values, oov),
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
                    DType::String | DType::Enum { .. } => {
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

    #[test]
    fn flatten_pooling_uses_configured_seq_len() {
        let spec = FeatureSpec {
            name: "seq".to_string(),
            vocab_size: 16,
            embed_dim: 4,
            pooling: PoolingStrategy::Flatten,
            seq_len: Some(3),
            truncation: TruncationSide::Head,
        };
        let col = vec![Fv::IntList(vec![1, 2]), Fv::IntList(vec![3, 4, 5, 6])];

        let tensor = feature_column_to_tensor(&spec, &col, 2, &Device::Cpu).unwrap();

        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn flatten_pooling_requires_seq_len() {
        let spec = FeatureSpec {
            name: "seq".to_string(),
            vocab_size: 16,
            embed_dim: 4,
            pooling: PoolingStrategy::Flatten,
            seq_len: None,
            truncation: TruncationSide::Head,
        };
        let col = vec![Fv::IntList(vec![1, 2])];

        let err = feature_column_to_tensor(&spec, &col, 1, &Device::Cpu).unwrap_err();

        assert!(err.contains("requires seq_len"));
    }

    #[test]
    fn test_feature_column_to_tensor_respects_truncation_side() {
        let spec_head = FeatureSpec {
            name: "seq".to_string(),
            vocab_size: 16,
            embed_dim: 4,
            pooling: PoolingStrategy::Mean,
            seq_len: Some(3),
            truncation: TruncationSide::Head,
        };
        let col = vec![Fv::IntList(vec![10, 20, 30, 40])];
        let tensor_head = feature_column_to_tensor(&spec_head, &col, 1, &Device::Cpu).unwrap();
        assert_eq!(
            tensor_head.to_vec2::<u32>().unwrap(),
            vec![vec![10, 20, 30]]
        );

        let spec_tail = FeatureSpec {
            name: "seq".to_string(),
            vocab_size: 16,
            embed_dim: 4,
            pooling: PoolingStrategy::Mean,
            seq_len: Some(3),
            truncation: TruncationSide::Tail,
        };
        let tensor_tail = feature_column_to_tensor(&spec_tail, &col, 1, &Device::Cpu).unwrap();
        assert_eq!(
            tensor_tail.to_vec2::<u32>().unwrap(),
            vec![vec![20, 30, 40]]
        );
    }

    #[test]
    fn test_predict_broadcast_correctness() {
        use crate::feats::config::FlowConfig;
        use crate::feats::dag::FeatureDag;
        use crate::models::lr::LogisticRegression;
        use candle_nn::VarMap;

        let yaml = r#"
version: 1.0.0
sources:
  - name: user_id
    source: User
    dtype: int
    default_val: '0'
  - name: item_id
    source: Item
    dtype: int
    default_val: '0'
operators:
  - name: user_hash
    op_type: FeatureHash
    inputs: [user_id]
    outputs: [user_idx]
    params: { vocab_size: 100 }
    embed: { vocab_size: 100, embed_dim: 8 }
  - name: item_hash
    op_type: FeatureHash
    inputs: [item_id]
    outputs: [item_idx]
    params: { vocab_size: 100 }
    embed: { vocab_size: 100, embed_dim: 8 }
"#;

        let flow_config = FlowConfig::from_yaml(yaml).unwrap();
        let dag = FeatureDag::from_config(flow_config, false, None).unwrap();

        let embed_specs: Vec<FeatureSpec> = dag
            .embeddable_features()
            .iter()
            .map(|(n, e)| FeatureSpec {
                name: n.to_string(),
                vocab_size: e.vocab_size,
                embed_dim: e.embed_dim,
                pooling: e.pooling,
                seq_len: e.seq_len,
                truncation: e.truncation,
            })
            .collect();

        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        let model = Box::new(LogisticRegression::new(vb, &embed_specs).unwrap());
        let engine = InferenceEngine::new(dag, model, embed_specs, device.clone());

        let mut user = HashMap::new();
        user.insert("user_id".to_string(), serde_json::json!(42));

        let mut item1 = HashMap::new();
        item1.insert("item_id".to_string(), serde_json::json!(101));

        let mut item2 = HashMap::new();
        item2.insert("item_id".to_string(), serde_json::json!(102));

        let items = vec![item1, item2];

        let (preds, _metrics) = engine.predict_broadcast(&user, &items).unwrap();
        assert_eq!(preds.len(), 2);
    }
}

fn normalize_enum_value(
    value: &str,
    values: &[String],
    oov: &Option<String>,
) -> Result<Fv, String> {
    if values.iter().any(|candidate| candidate == value) {
        return Ok(Fv::Str(value.to_string()));
    }
    if let Some(oov_value) = oov {
        return Ok(Fv::Str(oov_value.clone()));
    }
    Err(format!("unknown enum value '{}'", value))
}
