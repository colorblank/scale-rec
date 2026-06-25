# scale-rec

scale-rec 是一个推荐系统训练与推理框架。Python 侧负责数据处理、训练和 safetensors 权重导出；Rust 侧基于 Candle 加载同一份特征配置和模型权重，提供 HTTP 推理服务。

核心约定：

- 特征配置由 YAML 统一描述，Python 训练和 Rust 推理共享。
- 模型结构由 model config YAML 描述，权重由 Python 导出为 safetensors。
- 线上加载推荐使用 serving manifest，把权重、模型配置、特征配置、版本和校验信息绑定在一起。

## 文档导航

| 文档 | 内容 |
|---|---|
| [推荐排序系统教程](docs/tutorial/README.md) | 先建立全链路心智模型，再进入样本、特征、训练、发布、推理和排障 |
| [训练手册](docs/TRAINING_GUIDE.md) | 训练命令、数据格式、训练参数、checkpoint、发布 manifest、模型加载逻辑、压测 |
| [HTTP API](docs/API.md) | `/health`、`/models`、特征契约查询、`/predict`、`/predict/broadcast` 的请求、响应和错误格式 |
| [特征算子](docs/feature_operators.md) | 特征配置格式和 17 个算子说明，适合配合教程 03 章阅读 |
| [开发环境](docs/DEVELOPMENT.md) | Rust/Python 环境、常用命令、测试、格式化、端到端验证 |
| [Docker 打包](docker/README.md) | Linux 容器构建、运行环境变量、模型挂载方式 |
| [HTTP 压测报告](docs/http_benchmark_report.md) | GDCN+ESMM / UniMixer 压测结果和后端对比 |
| [设计改进记录](docs/design_improvements.md) | 当前架构评估和后续改进方向 |

## 代码结构

```text
scale-rec/
├── src/                            # Rust 推理引擎 + HTTP 服务
│   ├── feats/                      # FlowConfig、FeatureDag、特征算子
│   ├── layers/                     # Embedding、FM、MLP、Towers 等网络层
│   ├── models/                     # LR / DeepFM / MMoE / ESMM / GDCN+ESMM / UniMixer / TokenMixer-Large / RankMixer
│   ├── server/                     # InferenceEngine、ModelRegistry、HTTP routes
│   └── bin/                        # server、bench、demo_inference
├── python/
│   ├── src/train/                  # 训练 pipeline、模型、算子、artifact/manifest 管理
│   ├── src/scale_rec_demo/         # demo 数据生成和端到端验证脚本
│   ├── tests/                      # Python 测试
│   └── pyproject.toml              # Python 项目配置
├── examples/                       # discover 示例共享配置和模型配置
│   ├── gen_discover_config.py      # 生成 discover 特征配置的脚本
│   ├── models/                     # 按模型拆分的 model config（lr/gdcn_esmm/unimixer/token_mixer_large/rankmixer）
│   └── shared/                     # 共享的 feature / train / label 配置
├── docs/                           # 文档
├── docker/                         # Docker 打包入口
├── tests/                          # Rust 集成测试
└── Cargo.toml
```

## 快速开始

所有命令从仓库根目录执行。

### 1. 生成数据

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_discover_data \
  --label-policy examples/shared/discover_label_policy.yaml
```

### 2. 训练并发布模型

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm \
  --run-version 20260526_120000
```

多日训练文件可以用日期范围从 glob 结果中展开，日期取文件名中的第一个 8 位数字，闭区间内缺少任意一天会直接报错：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 --end-date 20260331 \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --init-weights python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  --epochs 3 --batch-size 1024 --no-header
```

`train_defaults.yaml` 负责训练默认值，`model_gdcn_esmm.yaml` 负责任务定义，`discover_label_policy.yaml` 只负责 demo 数据标签生成。三者职责分离，训练流程和评估指标都从配置读取，不再在代码里写死。`best.safetensors` 由 `eval.monitor_task`、`eval.monitor_metric` 和 `eval.monitor_mode` 明确决定；未指定 `monitor_task` 时使用模型 `tasks` 中的第一个任务。

`single`、`discover`、`all` 三个训练入口现在共享同一套特征预处理与可选预取逻辑；`examples/shared/train_defaults.yaml` 里的 `prefetch_batches` 可以用来控制后台提前准备多少个 batch，`checkpoint_interval_steps` / `checkpoint_interval_seconds` 可以控制训练中途的周期 checkpoint，`0` 表示关闭；`--resume-from` 可以从已有 checkpoint 恢复 model、optimizer、EMA、scheduler、step 和 epoch 状态。

当前 discover 示例模型配置包括：`examples/models/lr.yaml`、`gdcn_esmm.yaml`、`unimixer.yaml`、`token_mixer_large.yaml` 和 `rankmixer.yaml`。其中 UniMixer、TokenMixer-Large 和 RankMixer 都使用共享 `FeatureTokenizer` 把 DAG 输出特征投影为 token 序列。

训练启动后，日志会先打印一条数据摘要，包括总行数、训练/验证切分、batch_size、估算 batch 数、任务名和 label 映射。若出现 `No supervised batches were processed`，优先检查这条摘要里的 `labels` 和 `train/eval` 切分是否合理。

训练完成后会生成：

- `python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors`
- `python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml`
- `python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/configs/` 下的特征配置和模型配置副本
- `python/artifacts/demo/model_gdcn_esmm/20260526_120000/` 下的 run checkpoint、`run.manifest.yaml` 和 `embedding_bucket_report.yaml`

训练器会在所有实际执行反向传播的 batch 上累计每个 embedding bucket 的命中次数，
并把统计状态写入 checkpoint，断点续训后继续累计。最终发布时，`DictMapper`、
`FeatureHash`、`ParsedFeatureHash` 和 `ConcatHash` 的零命中 row 会在 serving 权重中
替换为该表活跃 row 的均值；训练 checkpoint 保持原始权重不变。

详细保存逻辑见 [训练手册 - 保存与推理导出](docs/TRAINING_GUIDE.md#保存与推理导出)。

### 3. 启动 HTTP 服务

推荐按 serving manifest 加载：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

只加载单个模型版本时，可以显式指定 manifest：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

旧的无 manifest `.safetensors` 产物仍可兼容加载，但必须提供 feature config fallback：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  --feature-config examples/shared/feature_config_discover.yaml
```

服务加载规则见 [训练手册 - 服务加载](docs/TRAINING_GUIDE.md#服务加载)，接口格式见 [HTTP API](docs/API.md)。

### 4. 调用接口

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm
curl http://127.0.0.1:8080/models/model_gdcn_esmm/features
```

Pointwise 推理：

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'
```

Broadcast 推理：

```bash
curl -X POST http://127.0.0.1:8080/predict/broadcast \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","user":{"user_id":42},"items":[{"item_id":500},{"item_id":501}]}'
```

## 模型发布与加载摘要

生产推荐加载 serving manifest，而不是直接依赖裸 `.safetensors` 文件。

serving manifest 记录：

- `model_id`、`model_version`、`model_type`
- `weights_file`、`feature_config_file`、`model_config_file`
- 权重、特征配置、模型配置的 sha256
- `weight_binding`
- `embedding_bucket_report_file`
- tasks、label mapping、metrics 等训练元数据

服务启动时：

- `--model-dir` 会递归扫描 serving manifest，跳过训练用的 `run.manifest.yaml`。
- `--model-path` 可重复传入，支持 manifest、目录或旧 `.safetensors`。
- 同一 `model_id` 可以加载多个 `model_version`。
- 未指定版本时，默认版本按版本字符串取最大值。
- 请求可通过 `version` 指定版本，通过 `fallback_version` 指定回退版本。

## 验证

常用开发验证命令：

```bash
cargo fmt
cargo check
cargo test

uvx --offline ruff check python/src/
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/ -v
```

端到端训练-导出-推理验证：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_lr,discover_gdcn_esmm,discover_unimixer,discover_token_mixer_large,discover_rankmixer --force-train
```

这条命令会串起 Python 训练、safetensors 导出、Rust 推理和输出比对。当前仓库已实测通过的主线是 `discover_lr`、`discover_gdcn_esmm`、`discover_unimixer`、`discover_token_mixer_large` 和 `discover_rankmixer`，其中所有输出项的最大差异都在浮点舍入误差范围内。

更多环境配置和命令见 [开发环境](docs/DEVELOPMENT.md)。
