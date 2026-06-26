# Reference

Reference 文档用于查稳定接口、配置项、文件格式和行为边界。

## Configuration

| Reference | Content |
|---|---|
| [CLI Reference](cli.md) | 训练、验证和服务启动命令参数 |
| [Feature Config Reference](feature_config.md) | feature config 顶层结构、sources、operators 和 embed |
| [Model Config Reference](model_config.md) | model YAML、output_contract、loss weighting 和 weight binding |
| [Feature operators](feature_operators.md) | 17 个特征算子的参数、输入输出和边界行为 |

## Serving

| Reference | Content |
|---|---|
| [HTTP API](http_api.md) | HTTP endpoints、请求、响应和错误格式 |
| [Prometheus metrics](prometheus_metrics.md) | `/metrics` 指标、计算逻辑和告警建议 |
| [Rust Model Loading](rust_model_loading.md) | manifest、model-dir、legacy safetensors 和版本选择 |

## Artifacts

| Reference | Content |
|---|---|
| [Artifact Reference](artifacts.md) | run directory、serving manifest、checkpoint 和 embedding bucket report |
| [产物发布与版本管理](../tutorials/artifact_publishing.md) | run manifest、serving manifest、sha256、model version 和回滚 |

## Development

| Reference | Content |
|---|---|
| [Development](development.md) | 本地开发、测试、格式化和一致性验证命令 |
| [Docker](../../docker/README.md) | 容器构建、运行环境变量和模型挂载方式 |
