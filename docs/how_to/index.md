# How-to Guides

How-to 文档面向具体操作：你已经知道目标，只需要稳定步骤和注意事项。当前详细内容主要仍在 [训练手册](../TRAINING_GUIDE.md) 和教程中，后续会逐步拆成独立页面。

## Training

| Task | Current documentation |
|---|---|
| 生成 demo 数据并训练模型 | [Getting Started](../getting_started.md)、[训练手册](../TRAINING_GUIDE.md#快速开始) |
| 使用独立验证集 | [Train with Independent Eval Data](train_with_independent_eval_data.md) |
| 多日训练和增量微调 | [Train with Multi-day Files](train_with_multi_day_files.md) |
| 从 checkpoint 恢复训练 | [训练手册](../TRAINING_GUIDE.md#保存与推理导出) |
| 调整多目标损失权重 | [Model Config Reference](../reference/model_config.md#loss-weighting)、[样本表、标签与任务定义](../tutorials/samples_labels_and_tasks.md) |

## Publishing and serving

| Task | Current documentation |
|---|---|
| 发布 safetensors 和 manifest | [Publish a Model](publish_model.md) |
| Rust 推理侧加载模型 | [Load a Model for Serving](load_model_for_serving.md) |
| 调用 HTTP API | [HTTP API](../reference/http_api.md) |
| 暴露和采集 Prometheus 指标 | [Prometheus metrics](../reference/prometheus_metrics.md) |

## Debugging and performance

| Task | Current documentation |
|---|---|
| 验证 Python/Rust 推理一致性 | [Getting Started: Verify Python and Rust consistency](../getting_started.md#verify-python-and-rust-consistency) |
| 排查 Python/Rust 输出不一致 | [Debug Python/Rust Mismatch](debug_python_rust_mismatch.md) |
| 查看 bucket hit count 和 feature quality | [Inspect Feature Quality](inspect_feature_quality.md) |
| 优化训练预处理性能 | [Tune Training Preprocessing](tune_training_preprocessing.md) |
| 运行 HTTP 压测 | [HTTP 压测报告](../notes/http_benchmark_report.md)、[训练手册](../TRAINING_GUIDE.md#http-压测) |
