# Notes

Notes 用于记录架构说明、性能分析、压测报告和设计演进。它们不一定是一步步操作指南，但有助于理解系统取舍。

## Architecture and design

| Note | Content |
|---|---|
| [Design improvements](design_improvements.md) | 当前架构评估、已完成拆分和后续改进方向 |

## Performance

| Note | Content |
|---|---|
| [HTTP benchmark report](http_benchmark_report.md) | GDCN+ESMM / UniMixer 压测结果、后端对比和压测命令 |
| [性能优化与大文件训练](../tutorials/performance_tuning.md) | pandas chunk、memory map、fast-no-na、prefetch 和真实 demo 输入压测 |

## Observability

| Note | Content |
|---|---|
| [Prometheus metrics](../reference/prometheus_metrics.md) | Rust 推理侧指标、计算逻辑和告警建议 |
| [训练评估与特征质量](../tutorials/evaluation_and_feature_quality.md) | feature quality、bucket utilization、inactive bucket 和训练侧质量报告 |
