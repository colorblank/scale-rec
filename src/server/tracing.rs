//! 请求耗时追踪：tracing spans 记录各阶段延迟。
use std::time::Instant;
use tracing::info;

/// 请求级别耗时记录。
pub struct RequestTimer {
    start: Instant,
    dag_us: u64,
    model_us: u64,
}

impl RequestTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            dag_us: 0,
            model_us: 0,
        }
    }

    pub fn record_dag(&mut self, us: u64) {
        self.dag_us = us;
    }
    pub fn record_model(&mut self, us: u64) {
        self.model_us = us;
    }

    /// 输出 span 日志。
    pub fn finish(self, model: &str, batch_size: usize) {
        let total_us = self.start.elapsed().as_micros() as u64;
        info!(
            model = model,
            batch = batch_size,
            total_us = total_us,
            dag_us = self.dag_us,
            model_us = self.model_us,
            "request complete"
        );
    }
}
