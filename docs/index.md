# scale-rec documentation

scale-rec 文档按 PyTorch 风格分为快速开始、教程、操作指南、参考手册和设计说明。先用 Getting Started 跑通链路，再按任务进入 Tutorials 或 How-to Guides；需要查参数、配置和接口时看 Reference。

## Get started

| 文档 | 内容 |
|---|---|
| [Getting Started](getting_started.md) | 从 demo 数据生成到训练、导出、HTTP 服务和一致性验证 |
| [Development](reference/development.md) | 本地环境、测试、格式化、端到端验证和开发注意事项 |

## Tutorials

教程按推荐系统工程链路组织，适合建立完整心智模型。

| 入口 | 内容 |
|---|---|
| [Tutorials index](tutorials/index.md) | PyTorch 风格教程入口，按推荐系统链路组织 |

## How-to guides

操作指南面向具体任务：训练、评估、发布、服务加载、排障和性能调优。

| 入口 | 内容 |
|---|---|
| [How-to index](how_to/index.md) | 常见操作入口和当前旧文档映射 |
| [Training guide](TRAINING_GUIDE.md) | 训练命令、数据格式、checkpoint、发布 manifest、服务加载和压测 |

## Reference

参考文档用于查接口、参数、配置和稳定行为。

| 入口 | 内容 |
|---|---|
| [Reference index](reference/index.md) | 配置、CLI、API、指标和 artifact 参考入口 |
| [CLI Reference](reference/cli.md) | 训练、验证和服务启动命令参数 |
| [Feature Config Reference](reference/feature_config.md) | feature config schema 和 embedding 配置 |
| [Model Config Reference](reference/model_config.md) | model YAML、output_contract 和权重绑定 |
| [Artifact Reference](reference/artifacts.md) | run 目录、serving manifest 和 bucket report |
| [Rust Model Loading](reference/rust_model_loading.md) | Rust 服务加载模型文件和版本选择规则 |
| [HTTP API](reference/http_api.md) | `/health`、`/models`、`/predict`、`/predict/broadcast` |
| [Prometheus metrics](reference/prometheus_metrics.md) | `/metrics` 指标、计算逻辑和告警建议 |
| [Feature operators](reference/feature_operators.md) | 17 个特征算子的参数、输入输出和边界行为 |

## Notes

说明文档记录架构、性能、压测和设计演进。

| 入口 | 内容 |
|---|---|
| [Notes index](notes/index.md) | 架构、性能和设计记录入口 |
| [HTTP benchmark report](notes/http_benchmark_report.md) | GDCN+ESMM / UniMixer 压测结果和后端对比 |
| [Design improvements](notes/design_improvements.md) | 当前架构评估和后续改进方向 |

## Architecture decisions

| ADR | 内容 |
|---|---|
| [ADR 0001: output_contract v1](adr/0001-output-contract-v1.md) | 统一模型输出、训练目标、指标和公开输出契约 |
