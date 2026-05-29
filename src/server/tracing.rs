//! 请求耗时追踪：tracing spans 记录各阶段延迟。
use std::time::Instant;
use tracing::debug;

use super::engine::InferenceMetrics;

/// 请求级别耗时记录。
pub struct RequestTimer {
    start: Instant,
    parse_us: u64,
    dag_us: u64,
    tensor_us: u64,
    forward_us: u64,
    response_us: u64,
}

impl RequestTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            parse_us: 0,
            dag_us: 0,
            tensor_us: 0,
            forward_us: 0,
            response_us: 0,
        }
    }

    pub fn record(&mut self, metrics: &InferenceMetrics) {
        self.parse_us = metrics.parse_us;
        self.dag_us = metrics.dag_us;
        self.tensor_us = metrics.tensor_us;
        self.forward_us = metrics.forward_us;
        self.response_us = metrics.response_us;
    }

    /// 输出 span 日志。
    pub fn finish(self, route: &str, model: &str, batch_size: usize) {
        let total_us = self.start.elapsed().as_micros() as u64;
        debug!(
            route = route,
            model = model,
            batch = batch_size,
            total_us = total_us,
            parse_us = self.parse_us,
            dag_us = self.dag_us,
            tensor_us = self.tensor_us,
            forward_us = self.forward_us,
            response_us = self.response_us,
            "request complete"
        );
    }
}
