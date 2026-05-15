//! DebugTracer：记录特征 DAG 管道中每个阶段的 I/O 值和异常。
use serde::Serialize;
use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// 阶段类型
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "UPPERCASE")]
pub enum StageType {
    DefaultInit,
    RawOverride,
    Operator,
}

/// 值快照：记录值和类型名（JSON 兼容）
#[derive(Debug, Clone, Serialize)]
pub struct ValueSnapshot {
    #[serde(rename = "v")]
    pub value: serde_json::Value,
    #[serde(rename = "t")]
    pub type_name: String,
}

impl ValueSnapshot {
    pub fn of(val: &crate::feats::ops::Fv) -> Self {
        use crate::feats::ops::Fv;
        match val {
            Fv::Int(i) => ValueSnapshot { value: serde_json::json!(*i), type_name: "int".into() },
            Fv::Float(f) => ValueSnapshot { value: serde_json::json!(*f), type_name: "float".into() },
            Fv::Str(s) => ValueSnapshot { value: serde_json::json!(s.clone()), type_name: "str".into() },
            Fv::StrList(l) => ValueSnapshot { value: serde_json::json!(l), type_name: "list[str]".into() },
            Fv::IntList(l) => ValueSnapshot { value: serde_json::json!(l), type_name: "list[int]".into() },
        }
    }
}

/// 异常记录
#[derive(Debug, Clone, Serialize)]
pub struct Anomaly {
    pub feature: String,
    pub reason: String,
}

/// 单个阶段追踪：记录一个算子的 I/O
#[derive(Debug, Clone, Serialize)]
pub struct StageTrace {
    #[serde(rename = "type")]
    pub stage_type: StageType,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub inputs: HashMap<String, ValueSnapshot>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub outputs: HashMap<String, ValueSnapshot>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub overridden: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub anomalies: Vec<Anomaly>,
}

impl StageTrace {
    pub fn defaults(outputs: HashMap<String, ValueSnapshot>) -> Self {
        StageTrace {
            stage_type: StageType::DefaultInit,
            name: String::new(),
            inputs: HashMap::new(),
            outputs,
            overridden: vec![],
            anomalies: vec![],
        }
    }
    pub fn overrides(overridden: Vec<String>, outputs: HashMap<String, ValueSnapshot>) -> Self {
        StageTrace {
            stage_type: StageType::RawOverride,
            name: String::new(),
            inputs: HashMap::new(),
            outputs,
            overridden,
            anomalies: vec![],
        }
    }
    pub fn operator(
        name: &str,
        inputs: HashMap<String, ValueSnapshot>,
        outputs: HashMap<String, ValueSnapshot>,
    ) -> Self {
        StageTrace {
            stage_type: StageType::Operator,
            name: name.to_string(),
            inputs,
            outputs,
            overridden: vec![],
            anomalies: vec![],
        }
    }
}

/// 单样本追踪
#[derive(Debug, Clone, Serialize)]
pub struct SampleTrace {
    pub sample: usize,
    pub stages: Vec<StageTrace>,
}

/// 调试配置
pub struct DebugConfig {
    pub max_trace_samples: usize,
    pub output_dir: String,
}

/// 调试追踪器：挂在 FeatureDag 上，在 execute() 各阶段记录 I/O。
pub struct DebugTracer {
    config: DebugConfig,
    traces: Mutex<Vec<SampleTrace>>,
    total_seen: Mutex<usize>,
    current: Mutex<Option<SampleTrace>>,
}

impl DebugTracer {
    pub fn new(config: DebugConfig) -> Self {
        DebugTracer {
            config,
            traces: Mutex::new(Vec::new()),
            total_seen: Mutex::new(0),
            current: Mutex::new(None),
        }
    }

    /// 开始追踪一个样本
    pub fn begin_sample(&self) {
        let mut seen = self.total_seen.lock().unwrap();
        if *seen >= self.config.max_trace_samples {
            *self.current.lock().unwrap() = None;
            *seen += 1;
            return;
        }
        *self.current.lock().unwrap() = Some(SampleTrace {
            sample: *seen,
            stages: vec![],
        });
        *seen += 1;
    }

    /// 记录默认值初始化阶段
    pub fn trace_defaults(&self, context: &HashMap<String, &crate::feats::ops::Fv>) {
        let mut curr = self.current.lock().unwrap();
        if let Some(ref mut t) = *curr {
            let outputs: HashMap<String, ValueSnapshot> = context
                .iter()
                .map(|(k, v)| (k.clone(), ValueSnapshot::of(*v)))
                .collect();
            t.stages.push(StageTrace::defaults(outputs));
        }
    }

    /// 记录 raw_inputs 覆盖阶段
    pub fn trace_overrides(
        &self,
        context: &HashMap<String, &crate::feats::ops::Fv>,
        overridden: Vec<String>,
    ) {
        let mut curr = self.current.lock().unwrap();
        if let Some(ref mut t) = *curr {
            let outputs: HashMap<String, ValueSnapshot> = context
                .iter()
                .map(|(k, v)| (k.clone(), ValueSnapshot::of(*v)))
                .collect();
            t.stages.push(StageTrace::overrides(overridden, outputs));
        }
    }

    /// 记录一个算子执行
    pub fn trace_operator(
        &self,
        op_name: &str,
        input_names: &[String],
        input_vals: &[&crate::feats::ops::Fv],
        output_names: &[String],
        output_val: &crate::feats::ops::Fv,
    ) {
        let mut curr = self.current.lock().unwrap();
        if let Some(ref mut t) = *curr {
            let inputs: HashMap<String, ValueSnapshot> = input_names
                .iter()
                .zip(input_vals.iter())
                .map(|(n, v)| (n.clone(), ValueSnapshot::of(*v)))
                .collect();
            let mut outputs = HashMap::new();
            let out_vs = ValueSnapshot::of(output_val);
            for name in output_names {
                outputs.insert(name.clone(), out_vs.clone());
            }
            t.stages
                .push(StageTrace::operator(op_name, inputs, outputs));
        }
    }

    /// 结束当前样本，存入 traces
    pub fn end_sample(&self) {
        let mut curr = self.current.lock().unwrap();
        if let Some(t) = curr.take() {
            self.traces.lock().unwrap().push(t);
        }
    }

    /// 保存 traces 和 summary 到 output_dir
    pub fn save(&self) {
        let dir = &self.config.output_dir;
        if dir.is_empty() {
            return;
        }
        let _ = fs::create_dir_all(dir);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros();

        // JSONL traces
        let tp = Path::new(dir).join(format!("traces_{}.jsonl", ts));
        let traces = self.traces.lock().unwrap();
        if let Ok(mut f) = fs::File::create(&tp) {
            for t in traces.iter() {
                if let Ok(line) = serde_json::to_string(t) {
                    let _ = writeln!(f, "{}", line);
                }
            }
        }

        // Summary JSON
        let sp = Path::new(dir).join(format!("summary_{}.json", ts));
        let summary = serde_json::json!({
            "total_samples": traces.len(),
        });
        if let Ok(mut f) = fs::File::create(&sp) {
            let _ = writeln!(
                f,
                "{}",
                serde_json::to_string_pretty(&summary).unwrap_or_default()
            );
        }

        println!("[Debug] summary -> {}", sp.display());
        println!(
            "[Debug] traces -> {} ({} samples)",
            tp.display(),
            traces.len()
        );
    }
}
