//! 特征监控指标：数据漂移检测、缺失率统计。

/// 在线特征统计（Welford 算法）。
pub struct FeatureMetrics {
    mean: f64,
    m2: f64,
    null_ratio: f64,
    count: usize,
    null_count: usize,
}

impl FeatureMetrics {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            null_ratio: 0.0,
            count: 0,
            null_count: 0,
        }
    }

    /// Welford 增量更新
    pub fn update(&mut self, val: f32, is_null: bool) {
        self.count += 1;
        if is_null {
            self.null_count += 1;
            self.null_ratio = self.null_count as f64 / self.count as f64;
        } else {
            self.null_ratio = self.null_count as f64 / self.count as f64;
            let x = val as f64;
            let delta = x - self.mean;
            self.mean += delta / (self.count - self.null_count) as f64;
            let delta2 = x - self.mean;
            self.m2 += delta * delta2;
        }
    }

    pub fn std_dev(&self) -> f64 {
        let n = (self.count - self.null_count).max(1) as f64;
        (self.m2 / n).sqrt()
    }

    pub fn report(&self) {
        tracing::info!(
            mean = self.mean,
            std_dev = self.std_dev(),
            null_ratio = self.null_ratio,
            "feature metrics report"
        );
    }
}

/// 性能埋点 (Performance Tracer)
pub struct PerformanceTracer;

impl PerformanceTracer {
    /// 记录节点耗时
    pub fn trace_execution_time(node_name: &str, duration_ms: f64) {
        tracing::info!(node_name, duration_ms, "feature node execution time");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford() {
        let mut m = FeatureMetrics::new();
        for v in [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            m.update(v, false);
        }
        assert!((m.mean - 5.0).abs() < 0.01);
        assert!(m.std_dev() > 0.0);
    }
}
