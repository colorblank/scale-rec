//! Prometheus 文本格式指标采集与导出。

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use super::engine::InferenceMetrics;
use super::registry::ModelRegistry;

const LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];
const BATCH_BUCKETS: &[f64] = &[
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Labels(Vec<(String, String)>);

impl Labels {
    fn new(values: &[(&str, &str)]) -> Self {
        Self(
            values
                .iter()
                .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
                .collect(),
        )
    }

    fn render(&self) -> String {
        let mut output = String::new();
        output.push('{');
        for (index, (key, value)) in self.0.iter().enumerate() {
            if index > 0 {
                output.push(',');
            }
            let _ = write!(output, "{key}=\"{}\"", escape_label(value));
        }
        output.push('}');
        output
    }
}

#[derive(Debug)]
struct Histogram {
    buckets: Vec<u64>,
    count: u64,
    sum: f64,
}

impl Histogram {
    fn new(bucket_count: usize) -> Self {
        Self {
            buckets: vec![0; bucket_count],
            count: 0,
            sum: 0.0,
        }
    }

    fn observe(&mut self, value: f64, bounds: &[f64]) {
        self.count += 1;
        self.sum += value;
        for (count, bound) in self.buckets.iter_mut().zip(bounds) {
            if value <= *bound {
                *count += 1;
            }
        }
    }
}

#[derive(Default)]
struct MetricsState {
    counters: HashMap<(&'static str, Labels), u64>,
    gauges: HashMap<(&'static str, Labels), i64>,
    histograms: HashMap<(&'static str, Labels), Histogram>,
}

/// 线程安全的 Prometheus 指标注册表。
pub struct PrometheusMetrics {
    state: Mutex<MetricsState>,
    started_at_seconds: u64,
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PrometheusMetrics {
    /// 创建空指标注册表。
    pub fn new() -> Self {
        Self {
            state: Mutex::new(MetricsState::default()),
            started_at_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_secs()),
        }
    }

    /// 记录 HTTP 请求状态和耗时。
    pub fn observe_http(&self, route: &str, method: &str, status: u16, duration_seconds: f64) {
        let mut state = lock_state(&self.state);
        let labels = Labels::new(&[
            ("route", route),
            ("method", method),
            ("status", &status.to_string()),
        ]);
        increment_counter(
            &mut state,
            "scale_rec_http_requests_total",
            labels.clone(),
            1,
        );
        observe_histogram(
            &mut state,
            "scale_rec_http_request_duration_seconds",
            labels,
            duration_seconds,
            LATENCY_BUCKETS,
        );
    }

    /// 增加指定路由和模型的进行中推理请求数。
    pub fn inference_started(&self, route: &str, model: &str) {
        add_gauge(
            &mut lock_state(&self.state),
            "scale_rec_inference_in_flight",
            Labels::new(&[("route", route), ("model", model)]),
            1,
        );
    }

    /// 减少指定路由和模型的进行中推理请求数。
    pub fn inference_finished(&self, route: &str, model: &str) {
        add_gauge(
            &mut lock_state(&self.state),
            "scale_rec_inference_in_flight",
            Labels::new(&[("route", route), ("model", model)]),
            -1,
        );
    }

    /// 记录推理请求结果、批大小和各阶段耗时。
    pub fn observe_inference(
        &self,
        route: &str,
        model: &str,
        status: &str,
        batch_size: usize,
        metrics: Option<&InferenceMetrics>,
    ) {
        let mut state = lock_state(&self.state);
        let request_labels = Labels::new(&[("route", route), ("model", model), ("status", status)]);
        increment_counter(
            &mut state,
            "scale_rec_inference_requests_total",
            request_labels,
            1,
        );
        observe_histogram(
            &mut state,
            "scale_rec_inference_batch_size",
            Labels::new(&[("route", route), ("model", model)]),
            batch_size as f64,
            BATCH_BUCKETS,
        );
        let Some(metrics) = metrics else {
            return;
        };
        for (stage, value_us) in [
            ("parse", metrics.parse_us),
            ("dag", metrics.dag_us),
            ("tensor", metrics.tensor_us),
            ("forward", metrics.forward_us),
            ("response", metrics.response_us),
        ] {
            observe_histogram(
                &mut state,
                "scale_rec_inference_stage_duration_seconds",
                Labels::new(&[("route", route), ("model", model), ("stage", stage)]),
                value_us as f64 / 1_000_000.0,
                LATENCY_BUCKETS,
            );
        }
        increment_counter(
            &mut state,
            "scale_rec_feature_default_values_total",
            Labels::new(&[("route", route), ("model", model)]),
            metrics.default_value_hits,
        );
        increment_counter(
            &mut state,
            "scale_rec_feature_empty_sequences_total",
            Labels::new(&[("route", route), ("model", model)]),
            metrics.empty_sequence_hits,
        );
    }

    /// 按 Prometheus text exposition format 导出当前指标。
    pub fn render(&self, registry: &ModelRegistry) -> String {
        let state = lock_state(&self.state);
        let mut output = String::new();
        write_metric_header(
            &mut output,
            "scale_rec_build_info",
            "Scale-rec build information.",
            "gauge",
        );
        let _ = writeln!(
            output,
            "scale_rec_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        );
        write_metric_header(
            &mut output,
            "scale_rec_process_start_time_seconds",
            "Unix timestamp when the process metrics registry was created.",
            "gauge",
        );
        let _ = writeln!(
            output,
            "scale_rec_process_start_time_seconds {}",
            self.started_at_seconds
        );

        render_counter(
            &mut output,
            &state,
            "scale_rec_http_requests_total",
            "Total HTTP requests grouped by route, method, and status.",
        );
        render_histogram(
            &mut output,
            &state,
            "scale_rec_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            LATENCY_BUCKETS,
        );
        render_counter(
            &mut output,
            &state,
            "scale_rec_feature_default_values_total",
            "Total feature values filled from configured defaults.",
        );
        render_counter(
            &mut output,
            &state,
            "scale_rec_feature_empty_sequences_total",
            "Total empty feature sequences observed in inference requests.",
        );
        render_counter(
            &mut output,
            &state,
            "scale_rec_inference_requests_total",
            "Total inference requests grouped by route, model, and result status.",
        );
        render_gauge(
            &mut output,
            &state,
            "scale_rec_inference_in_flight",
            "Current in-flight inference requests.",
        );
        render_histogram(
            &mut output,
            &state,
            "scale_rec_inference_batch_size",
            "Inference request batch size.",
            BATCH_BUCKETS,
        );
        render_histogram(
            &mut output,
            &state,
            "scale_rec_inference_stage_duration_seconds",
            "Inference stage latency in seconds.",
            LATENCY_BUCKETS,
        );
        drop(state);

        write_metric_header(
            &mut output,
            "scale_rec_model_loaded",
            "Whether a model version is currently loaded.",
            "gauge",
        );
        for model in registry.list_info() {
            for version in model.versions {
                let _ = writeln!(
                    output,
                    "scale_rec_model_loaded{{model=\"{}\",version=\"{}\",model_type=\"{}\",is_default=\"{}\"}} 1",
                    escape_label(&model.name),
                    escape_label(&version.version),
                    escape_label(&version.model_type),
                    version.is_default
                );
            }
        }
        output
    }
}

fn lock_state(mutex: &Mutex<MetricsState>) -> std::sync::MutexGuard<'_, MetricsState> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn increment_counter(state: &mut MetricsState, name: &'static str, labels: Labels, value: u64) {
    *state.counters.entry((name, labels)).or_default() += value;
}

fn add_gauge(state: &mut MetricsState, name: &'static str, labels: Labels, delta: i64) {
    let gauge = state.gauges.entry((name, labels)).or_default();
    *gauge = (*gauge + delta).max(0);
}

fn observe_histogram(
    state: &mut MetricsState,
    name: &'static str,
    labels: Labels,
    value: f64,
    bounds: &[f64],
) {
    state
        .histograms
        .entry((name, labels))
        .or_insert_with(|| Histogram::new(bounds.len()))
        .observe(value, bounds);
}

fn render_counter(output: &mut String, state: &MetricsState, name: &str, help: &str) {
    write_metric_header(output, name, help, "counter");
    for ((metric_name, labels), value) in &state.counters {
        if *metric_name == name {
            let _ = writeln!(output, "{name}{} {value}", labels.render());
        }
    }
}

fn render_gauge(output: &mut String, state: &MetricsState, name: &str, help: &str) {
    write_metric_header(output, name, help, "gauge");
    for ((metric_name, labels), value) in &state.gauges {
        if *metric_name == name {
            let _ = writeln!(output, "{name}{} {value}", labels.render());
        }
    }
}

fn render_histogram(
    output: &mut String,
    state: &MetricsState,
    name: &str,
    help: &str,
    bounds: &[f64],
) {
    write_metric_header(output, name, help, "histogram");
    for ((metric_name, labels), histogram) in &state.histograms {
        if *metric_name != name {
            continue;
        }
        for (bound, count) in bounds.iter().zip(&histogram.buckets) {
            let mut bucket_labels = labels.0.clone();
            bucket_labels.push(("le".to_string(), bound.to_string()));
            let _ = writeln!(
                output,
                "{name}_bucket{} {}",
                Labels(bucket_labels).render(),
                count
            );
        }
        let mut infinite_labels = labels.0.clone();
        infinite_labels.push(("le".to_string(), "+Inf".to_string()));
        let _ = writeln!(
            output,
            "{name}_bucket{} {}",
            Labels(infinite_labels).render(),
            histogram.count
        );
        let _ = writeln!(output, "{name}_sum{} {}", labels.render(), histogram.sum);
        let _ = writeln!(
            output,
            "{name}_count{} {}",
            labels.render(),
            histogram.count
        );
    }
}

fn write_metric_header(output: &mut String, name: &str, help: &str, metric_type: &str) {
    let _ = writeln!(output, "# HELP {name} {help}");
    let _ = writeln!(output, "# TYPE {name} {metric_type}");
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_counters_histograms_and_escaped_labels() {
        let metrics = PrometheusMetrics::new();
        metrics.observe_http("/predict", "POST", 200, 0.01);
        metrics.observe_inference(
            "/predict",
            "ranker\"blue",
            "ok",
            8,
            Some(&InferenceMetrics {
                parse_us: 100,
                dag_us: 200,
                tensor_us: 300,
                forward_us: 400,
                response_us: 500,
                default_value_hits: 2,
                empty_sequence_hits: 1,
            }),
        );

        let registry = ModelRegistry::from_model_dir(std::env::temp_dir().as_path()).unwrap();
        let output = metrics.render(&registry);

        assert!(output.contains("scale_rec_http_requests_total"));
        assert!(output.contains("scale_rec_inference_batch_size_bucket"));
        assert!(output.contains("model=\"ranker\\\"blue\""));
        assert!(output.contains("stage=\"forward\""));
        assert!(output.contains("scale_rec_feature_default_values_total"));
    }
}
