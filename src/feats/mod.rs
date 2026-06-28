//! 特征预处理引擎：通过 scale-rec-core 提供配置、DAG 构建、算子库，本地提供 DAG facade、监控、调试。
pub use scale_rec_core::feats::*;

/// 特征 DAG 执行引擎：拓扑排序、算子调度、样本执行。
pub mod dag;
/// 监控指标：数据漂移检测、缺失率统计、性能埋点。
pub mod metrics;
/// 调试追踪：逐阶段记录特征处理的输入/输出。
pub mod debug;
