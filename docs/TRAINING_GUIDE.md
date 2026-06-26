# 训练手册

本文档保留原 `TRAINING_GUIDE.md` 路径，作为训练、发布和服务加载的兼容导航页。新的文档结构按 PyTorch 风格拆分为：

- [Getting Started](getting_started.md)：最短可运行路径。
- [How-to Guides](how_to/index.md)：具体操作步骤。
- [Reference](reference/index.md)：CLI、配置、artifact、模型加载和 API 参考。
- [Tutorials](tutorials/index.md)：按推荐系统链路学习。

如果你只想跑通一次训练，直接看 [Getting Started](getting_started.md)。

## 教程对照

| 主题 | 新位置 |
|---|---|
| 离线训练流程 | [Offline Training](tutorials/offline_training.md) |
| 多日训练与增量微调 | [Multi-day Training](tutorials/multi_day_training.md)、[Train with Multi-day Files](how_to/train_with_multi_day_files.md) |
| 模型结构与权重绑定 | [Model Structure and Weight Binding](tutorials/model_and_weight_binding.md) |
| 训练评估与特征质量 | [Evaluation and Feature Quality](tutorials/evaluation_and_feature_quality.md)、[Inspect Feature Quality](how_to/inspect_feature_quality.md) |
| 产物发布与版本管理 | [Artifact Publishing](tutorials/artifact_publishing.md)、[Artifact Reference](reference/artifacts.md) |
| Rust 在线推理服务 | [Rust Inference Service](tutorials/rust_inference_service.md)、[Rust Model Loading](reference/rust_model_loading.md) |
| 性能优化与大文件训练 | [Performance Tuning](tutorials/performance_tuning.md)、[Tune Training Preprocessing](how_to/tune_training_preprocessing.md) |
| Debug 与一致性验证 | [Debugging and Consistency](tutorials/debugging_consistency.md)、[Debug Python/Rust Mismatch](how_to/debug_python_rust_mismatch.md) |

## 阅读顺序

| 目标 | 推荐阅读 |
|---|---|
| 先跑起来 | [Getting Started](getting_started.md) |
| 查命令参数 | [CLI Reference](reference/cli.md) |
| 查 feature config | [Feature Config Reference](reference/feature_config.md) |
| 查 model config / output_contract | [Model Config Reference](reference/model_config.md) |
| 查 artifact / manifest | [Artifact Reference](reference/artifacts.md) |
| 查 Rust 加载模型规则 | [Rust Model Loading](reference/rust_model_loading.md) |
| 查 HTTP 请求响应 | [HTTP API](reference/http_api.md) |
| 查 Prometheus 指标 | [Prometheus metrics](reference/prometheus_metrics.md) |
| 查特征算子 | [Feature operators](reference/feature_operators.md) |

## 快速开始

最短路径：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_discover_data \
  --label-policy examples/shared/discover_label_policy.yaml

PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm

PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

完整说明见 [Getting Started](getting_started.md)。

## 训练流程

训练链路由四类配置组成：

| 配置 | 说明 |
|---|---|
| `examples/shared/feature_config_discover.yaml` | 原始字段、DAG、embedding、label role |
| `examples/shared/discover_label_policy.yaml` | demo 标签生成规则，不参与模型前向 |
| `examples/shared/train_defaults.yaml` | batch size、optimizer、eval、checkpoint、EMA 等默认值 |
| `examples/models/*.yaml` | 模型结构、output_contract、loss、metrics、outputs |

训练主流程：

1. pandas chunk 读取样本。
2. Python feature DAG 预处理 batch。
3. 构造模型输入 tensor。
4. PyTorch 模型前向。
5. 按 `output_contract.objectives` 计算 loss。
6. 反向传播并更新参数。
7. 评估 metrics。
8. 导出 checkpoint、serving 权重和 manifest。

教程见 [Offline Training](tutorials/offline_training.md)。

## 数据格式

discover demo 默认是无 header TSV，列顺序来自 feature config 的 `sources`。

常用数据参数见 [CLI Reference](reference/cli.md#data-arguments)。

独立验证集：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data data/train.tsv \
  --eval-data data/eval.tsv \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --no-header
```

验证文件必须与训练文件格式、字段和字段顺序一致。详细步骤见 [Train with Independent Eval Data](how_to/train_with_independent_eval_data.md)。

多日训练见 [Train with Multi-day Files](how_to/train_with_multi_day_files.md)。

## 特征配置

feature config 是 Python 训练和 Rust 推理共享的特征契约。它定义：

- `data_sources`
- `sources`
- `operators`
- operator 输出上的 `embed`
- `role: feature | label | discard`

配置参考见 [Feature Config Reference](reference/feature_config.md)，算子参考见 [Feature operators](reference/feature_operators.md)。

## 特征预处理 Debug

排查顺序：

1. 先确认训练/推理使用同一份 feature config。
2. 对单行样本执行 Python DAG，观察 source/default/operator 输出。
3. 对同一行执行 Rust DAG 或 `scale_rec_demo.verify_all`。
4. 比较 DictMapper、FeatureHash、Split、sequence padding/truncation。
5. 确认中文、分隔符、NULL 标记和文件解析没有造成列错位。

详细步骤见 [Debug Python/Rust Mismatch](how_to/debug_python_rust_mismatch.md) 和 [Debugging and Consistency](tutorials/debugging_consistency.md)。

## 模型配置

model config YAML 定义模型结构和任务语义。当前示例模型统一使用 `output_contract.version: 1`。

参考：

- [Model Config Reference](reference/model_config.md)
- [Model Structure and Weight Binding](tutorials/model_and_weight_binding.md)
- [ADR 0001: output_contract v1](adr/0001-output-contract-v1.md)

## 训练参数

训练参数集中见 [CLI Reference](reference/cli.md)。

常用入口：

- [Data arguments](reference/cli.md#data-arguments)
- [Config arguments](reference/cli.md#config-arguments)
- [Artifact arguments](reference/cli.md#artifact-arguments)

`output_contract` 路径只支持 static loss weighting，权重来自 `objectives[].weight`。说明见 [Model Config Reference: Loss weighting](reference/model_config.md#loss-weighting)。

## 训练技巧

常用建议：

- 大文件训练优先调 `--read-chunk-rows`、`--fast-no-na`、`--memory-map` 和 prefetch。
- 多日文件用 `--data-glob` + 日期闭区间，缺失日期直接失败。
- 从已有模型继续训练用 `--init-weights`。
- 中断恢复用 `--resume-from`。
- 改特征、模型结构或权重命名后必须跑 Python/Rust 一致性验证。

性能调优见 [Tune Training Preprocessing](how_to/tune_training_preprocessing.md)。

## 评估监控

评估指标来自 model config 的 `output_contract.metrics`。feature quality 和 bucket hit count 用于发现：

- source 缺失率。
- 默认值命中率。
- sequence 空值率和 padding rate。
- embedding bucket utilization。
- DictMapper default hit。

详细说明见 [Evaluation and Feature Quality](tutorials/evaluation_and_feature_quality.md) 和 [Inspect Feature Quality](how_to/inspect_feature_quality.md)。

## 保存与推理导出

默认发布目录：

```text
serving/
├── model.manifest.yaml
├── model.safetensors
└── configs/
    ├── feature_config.yaml
    └── model_config.yaml
```

文件职责见 [Artifact Reference](reference/artifacts.md)，发布操作见 [Publish a Model](how_to/publish_model.md)。

发布 serving 权重时，零命中 embedding row 会根据完整训练流 bucket report 替换为活跃 row 均值；训练 checkpoint 保持原始权重。详细规则见 [Artifact Reference: Embedding bucket report](reference/artifacts.md#embedding-bucket-report)。

### 服务加载

推荐按 manifest 或 model directory 加载：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo
```

单模型版本：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

加载规则见 [Rust Model Loading](reference/rust_model_loading.md)，操作步骤见 [Load a Model for Serving](how_to/load_model_for_serving.md)。

## HTTP 压测

HTTP 压测应区分 synthetic smoke 和真实 discover 输入压测。真实压测要传入 discover TSV 和 feature config。

```bash
cargo run --bin bench --release -- \
  --url http://127.0.0.1:8080 \
  --model model_gdcn_esmm \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml
```

说明见 [Performance Tuning](tutorials/performance_tuning.md) 和 [HTTP benchmark report](notes/http_benchmark_report.md)。

## 代码架构

主要目录：

```text
python/src/train/
├── app/         # CLI、入口、artifact/manifest 管理
├── core/        # config、DAG、executor、preprocessor、output_contract
├── models/      # PyTorch 模型
├── ops/         # Python 特征算子
└── training/    # trainer、loss、metrics、quality

src/
├── feats/       # Rust 特征配置、DAG 和算子
├── models/      # Candle 模型
└── server/      # HTTP serving
```

开发命令见 [Development](reference/development.md)。
