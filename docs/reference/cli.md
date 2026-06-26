# CLI Reference

本文档是命令行参数的唯一完整参考。教程和 How-to 只保留常用命令示例；需要查参数、默认值和约束时以本文档为准。

所有命令从仓库根目录执行。Python 命令统一使用 `uv run --project python`，并设置 `PYTHONPATH=python/src:$PYTHONPATH`。

## Entrypoints

| Entry point | Purpose | 参数表 |
|---|---|---|
| `python -m scale_rec_demo.generate_demo_data` | 生成 demo TSV | [Generate demo data](#generate-demo-data) |
| `python -m train.app.main single` | 单模型 CSV/Parquet 训练 | [Train: single](#train-single) |
| `python -m train.app.main demo` | demo TSV 训练 | [Train: demo](#train-demo) |
| `python -m train.app.main all` | 在同一数据集上批量训练示例模型 | [Train: all](#train-all) |
| `python -m train.main <mode>` | 兼容训练入口，转发到 `train.app.main` | 参数同 `train.app.main` |
| `python -m scale_rec_demo.verify_all` | Python 训练导出与 Rust 推理一致性验证 | [Verify all](#verify-all) |
| `python -m scale_rec_demo.check_weight_bindings` | 检查 PyTorch state_dict key 与 Rust Candle 路径绑定 | [Check weight bindings](#check-weight-bindings) |
| `cargo run --bin server --release -- ...` | Rust HTTP 推理服务 | [Rust server](#rust-server) |
| `cargo run --bin bench --release -- ...` | HTTP 压测工具 | [Rust bench](#rust-bench) |
| `cargo run --bin demo_inference --release -- ...` | 离线加载权重并对 CSV 推理 | [Rust demo inference](#rust-demo-inference) |
| `cargo run --bin validate_manifest --release -- ...` | 校验 serving manifest 与权重绑定 | [Rust validate manifest](#rust-validate-manifest) |

## Generate demo data

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_demo_data \
  --label-policy examples/shared/demo_label_policy.yaml
```

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--label-policy` | path | `examples/shared/demo_label_policy.yaml` | demo 标签生成规则 YAML | 必须可读 |

输出路径固定为 `python/artifacts/demo/demo_train_data.txt`。当前命令没有 `--output` 参数。

## Train command overview

训练入口格式：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main <single|demo|all> --run-name <name> [args...]
```

三个训练子命令共享 data、training、artifact 和 runtime 参数。没有在 CLI 显式传入的训练参数会从 `--train-config` 读取；默认配置文件是 `examples/shared/train_defaults.yaml`。

默认训练配置：

| 配置项 | 默认 |
|---|---:|
| `epochs` | `30` |
| `batch_size` | `64` |
| `prefetch_batches` | `2` |
| `eval_samples` | `400` |
| `eval_interval` | `50` |
| `log_interval` | `10` |
| `optim.name` | `adamw` |
| `optim.lr` | `0.005` |
| `optim.weight_decay` | `0.0001` |
| `lr_schedule.warmup_steps` | `200` |
| `lr_schedule.min_lr_ratio` | `0.01` |
| `grad_max_norm` | `1.0` |
| `early_stopping_patience` | `5` |
| `ema_decay` | `0.999` |
| `loss_weighting` | `static` |
| `eval.metrics` | `auc` |
| `eval.monitor_metric` | `auc` |
| `eval.monitor_mode` | `auto` |
| `eval.gauc_group_feature` | `user_id` |

### Common data arguments

适用于 `single`、`demo`、`all`。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--data` | path | 空 | 单文件训练数据路径 | 未传 `--data-glob` 时必填 |
| `--eval-data` | path | 空 | 独立验证文件 | 必须与训练文件格式、字段和字段顺序完全一致 |
| `--data-glob` | glob | 空 | 多日文件 glob；设置后优先于 `--data` | 文件名必须包含 `YYYYMMDD` |
| `--start-date` | string | 空 | `--data-glob` 闭区间开始日期 | 使用 `--data-glob` 时必填，格式 `YYYYMMDD` |
| `--end-date` | string | 空 | `--data-glob` 闭区间结束日期 | 使用 `--data-glob` 时必填，格式 `YYYYMMDD`，且 `start <= end` |

### Common training arguments

适用于 `single`、`demo`、`all`。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--train-config` | path | `examples/shared/train_defaults.yaml` | 训练默认配置 YAML | 文件不存在时使用代码内置默认 |
| `--epochs` | int | 来自 `--train-config` | 训练 epoch 数 | 正整数 |
| `--batch-size` | int | 来自 `--train-config` | batch size | 正整数 |
| `--lr` | float | 来自 `--train-config` | 主 optimizer 学习率 | 正数 |
| `--weight-decay` | float | 来自 `--train-config` | 主 optimizer weight decay | 非负 |
| `--optim` | enum | 来自 `--train-config` | optimizer 类型 | `adamw` / `adam` / `sgd` |
| `--emb-lr` | float | 来自 `--train-config` | embedding 参数组学习率 | 空表示沿用主学习率 |
| `--emb-weight-decay` | float | 来自 `--train-config` | embedding 参数组 weight decay | 空表示沿用主 weight decay |
| `--eval-samples` | int | 来自 `--train-config` | 评估样本数或评估截断规模 | 非负 |
| `--eval-interval` | int | 来自 `--train-config` | step 级评估间隔 | 非负；0 表示关闭 step 间隔评估 |
| `--log-interval` | int | 来自 `--train-config` | step 级日志间隔 | 正整数 |
| `--prefetch-batches` | int | 来自 `--train-config` | 训练预处理 prefetch batch 数 | 非负 |
| `--checkpoint-interval-steps` | int | 来自 `--train-config` | 按 step 保存 periodic checkpoint | 0 表示关闭 |
| `--checkpoint-interval-seconds` | float | 来自 `--train-config` | 按时间保存 periodic checkpoint | 0 表示关闭 |
| `--warmup-steps` | int | 来自 `--train-config` | 学习率 warmup steps | 非负 |
| `--min-lr-ratio` | float | 来自 `--train-config` | 学习率调度最低比例 | 通常在 `[0, 1]` |
| `--grad-max-norm` | float | 来自 `--train-config` | 梯度裁剪范数 | 0 或空表示不裁剪 |
| `--early-stopping` | int | 来自 `--train-config` | early stopping patience | 非负 |
| `--no-ema` | flag | false | 禁用 EMA | 设置后 `ema_decay=0.0` |
| `--ema-decay` | float | 来自 `--train-config` | EMA decay | 通常在 `[0, 1)` |
| `--loss-weighting` | enum | 来自 `--train-config` | legacy 多任务 loss weighting | `equal` / `static` / `uncertainty`；原生 `output_contract` 使用 objectives[].weight |
| `--tb-dir` | path | 来自 `--train-config` | TensorBoard 输出目录 | 空表示不写 |
| `--eval-metrics` | csv | 来自 `--train-config` | 评估指标列表 | 逗号分隔，例如 `auc,logloss` |
| `--monitor-metric` | string | 来自 `--train-config` | early stopping / best checkpoint 监控指标 | 需存在于评估输出 |
| `--monitor-task` | string | 来自 `--train-config` | 监控任务名 | 空表示自动选择 |
| `--monitor-mode` | enum | 来自 `--train-config` | 监控方向 | `auto` / `max` / `min` |
| `--eval-log` | path | 来自 `--train-config` | 评估日志输出路径 | 空表示不写 |
| `--gauc-group-feature` | string | 来自 `--train-config` | GAUC 分组特征 | 需存在于 batch values |
| `--init-weights` | path | 空 | 用 safetensors 初始化模型参数 | 不能与 `--resume-from` 同时使用 |
| `--resume-from` | path | 空 | 从 checkpoint 权重或 `.resume.pt` sidecar 恢复训练 | 不能与 `--init-weights` 同时使用 |

### Common artifact arguments

适用于 `single`、`demo`、`all`。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--artifact-dir` / `--export-dir` | path | `python/artifacts/demo` | run 目录根路径 | 可创建或已存在 |
| `--publish-path` / `--export-path` | path | 自动生成 | 最终 serving 权重路径 | 未传时由 artifact manager 生成 |
| `--model-name` | string | 自动推导 | 逻辑模型名，写入 manifest | 建议稳定、可读 |
| `--run-name` | string | 必填 | 人类可读的 run 名称，用于日志文件命名 | 必须显式传入；会被规范化为文件名前缀 |
| `--run-version` | string | 自动生成 | run/version 字符串 | 建议使用可排序时间戳 |
| `--keep-checkpoints` | int | `3` | 保留 checkpoint 数 | 非负 |

### Common runtime and logging arguments

适用于 `single`、`demo`、`all`。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--device` | enum | `auto` | 训练设备 | `auto` / `cpu` / `cuda` / `mps` |
| `--log-level` | enum | `INFO` | 控制台日志级别 | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--file-log-level` | enum | `DEBUG` | 文件日志级别 | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--log-dir` | path | 空；训练入口通常使用 `<artifact-dir>/logs` | 自动生成时间戳日志文件的目录 | 被 `--log-file` 覆盖 |
| `--log-file` | path | 空 | 显式日志文件路径 | 优先级高于 `--log-dir` |

## Train: single

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main single \
  --data data/train.csv \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/lr.yaml \
  --run-name single_train
```

`single` 适合 CSV/Parquet 单模型训练。它支持 [common data arguments](#common-data-arguments)、[common training arguments](#common-training-arguments)、[common artifact arguments](#common-artifact-arguments) 和 [common runtime and logging arguments](#common-runtime-and-logging-arguments)。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--feature-config` | path | `examples/shared/feature_config_demo.yaml` | 特征 DAG 配置 | 必须可读 |
| `--model-config` | path | 必填 | 模型结构与 `output_contract` 配置 | 必须可读 |
| `--debug` | int | `0` | FeatureDag debug 开关 | `>0` 时启用 debug mode |

## Train: demo

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name demo_train \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400
```

`demo` 适合 demo-main-sort TSV 训练。它支持 [common data arguments](#common-data-arguments)、[common training arguments](#common-training-arguments)、[common artifact arguments](#common-artifact-arguments) 和 [common runtime and logging arguments](#common-runtime-and-logging-arguments)。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--feature-config` | path | `examples/shared/feature_config_demo.yaml` | 特征 DAG 配置 | 必须可读 |
| `--model-config` | path | 必填 | 模型结构与 `output_contract` 配置 | 必须可读 |
| `--no-header` | flag | false | 输入 TSV 无 header 行 | demo 数据需要开启 |
| `--null-markers` | list | `NULL \N null None ""` | 识别为空值的字符串集合 | 空字符串也属于默认集合 |
| `--separator` | string | `\t` | 字段分隔符 | pandas `read_csv(sep=...)` 使用 |
| `--read-chunk-rows` | int | `0` | pandas chunk 行数 | 0 表示使用基于 batch size 的默认值 |
| `--fast-no-na` | flag | false | 关闭 pandas NA 检测以提升读取速度 | 仅在后续默认值处理能覆盖缺失值时使用 |
| `--memory-map` | flag | false | 启用 pandas memory map | 仅适合本地未压缩文件 |

## Train: all

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main all \
  --data python/artifacts/demo/demo_train_data.txt \
  --run-name all_train \
  --models all
```

`all` 在同一数据集上批量训练多个示例模型。它支持 [common data arguments](#common-data-arguments)、[common training arguments](#common-training-arguments)、[common artifact arguments](#common-artifact-arguments) 和 [common runtime and logging arguments](#common-runtime-and-logging-arguments)。

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--feature-config` | path | `examples/shared/feature_config_demo.yaml` | 特征 DAG 配置 | 必须可读 |
| `--models` | csv 或 `all` | `all` | 要训练的模型 key 列表 | 逗号分隔；支持代码中注册的 demo key |
| `--model-config-demo-gdcn-esmm` | path | `examples/models/gdcn_esmm.yaml` | `demo_gdcn_esmm` 的模型配置 | 必须可读 |
| `--model-config-demo-unimixer` | path | `examples/models/unimixer.yaml` | `demo_unimixer` 的模型配置 | 必须可读 |
| `--debug` | int | `0` | FeatureDag debug 开关 | `>0` 时启用 debug mode |

## Verify all

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--models` | csv 或 `all` | `all` | 要验证的模型 key 列表 | 支持 `demo_lr`、`demo_deepfm`、`demo_mmoe`、`demo_esmm`、`demo_gdcn_esmm`、`demo_unimixer`、`demo_token_mixer_large`、`demo_rankmixer` |
| `--force-train` | flag | false | 强制重新训练并覆盖 demo 权重 | 会增加运行时间 |
| `--threshold` | float | `1e-4` | Python/Rust 输出最大允许差异 | 超过阈值视为失败 |

## Check weight bindings

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.check_weight_bindings --models all
```

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--models` | csv 或 `all` | `all` | 要检查的模型 key 列表 | 同 [Verify all](#verify-all) |
| `--feature-config` | path | `examples/shared/feature_config_demo.yaml` | 用于构建 feature info 的特征配置 | 必须可读 |
| `--keep-temp` | flag | false | 保留临时 safetensors 和 manifest | 用于排查 key/shape 问题 |

## Rust server

按 manifest 目录加载：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

只加载单个 manifest：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

| 参数 | 类型 | 默认 | 环境变量 | 说明 | 约束 |
|---|---|---|---|---|---|
| `--model-dir` | path | 空 | 无 | 模型目录；服务会扫描其中的 manifest | `--model-dir` 或 `--model-path` 至少传一个 |
| `--model-path` / `--model-manifest` | path，可重复 | 空 | 无 | 显式加载一个或多个 serving manifest | 可重复传入 |
| `--feature-config` | path | 空 | 无 | legacy loose safetensors 加载的 feature config fallback | manifest-driven 加载通常不需要 |
| `--port` | int | `8080` | `SCALE_REC_PORT` | HTTP 监听端口 | 无效值回退到 `8080` |
| `--worker-threads` | int | Tokio 默认 | 无 | Tokio worker threads | 解析失败时忽略 |
| `--blocking-threads` | int | Tokio 默认 | 无 | Tokio max blocking threads | 解析失败时忽略 |
| `--allowed-origin` | string，可重复 | localhost/127.0.0.1 常用前端端口 | `SCALE_REC_ALLOWED_ORIGINS` | CORS allow origin | 环境变量使用逗号分隔；CLI 可重复追加 |
| `--max-body-bytes` | int | `8388608` | `SCALE_REC_MAX_BODY_BYTES` | 请求体上限 | 必须大于 0 |
| `--rate-limit-per-second` | int | `1000` | `SCALE_REC_RATE_LIMIT_PER_SECOND` | 全局每秒请求数限制 | 必须大于 0 |
| `--max-concurrency` | int | `512` | `SCALE_REC_MAX_CONCURRENCY` | 最大并发请求数 | 必须大于 0 |
| `--request-timeout-secs` | int | `30` | `SCALE_REC_REQUEST_TIMEOUT_SECS` | 单请求超时秒数 | 必须大于 0 |

更多加载规则见 [Rust Model Loading](rust_model_loading.md)。

## Rust bench

```bash
cargo run --bin bench --release -- \
  --target http://127.0.0.1:8080 \
  --model model_gdcn_esmm \
  --mode broadcast \
  --input-file python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --no-header \
  --target-qps 300
```

| 参数 | 类型 | 默认 | 说明 | 约束 |
|---|---|---|---|---|
| `--target` | URL | `http://localhost:8080` | 服务 base URL | 不包含 endpoint；工具自动拼 `/predict` 或 `/predict/broadcast` |
| `--model` | string | `lr` | 请求体中的模型名 | 必须是服务已加载模型 |
| `--mode` | string | `pointwise` | 请求模式 | `broadcast` 使用 `/predict/broadcast`；其他值按 pointwise 处理 |
| `--concurrency` | int | `100` | closed-loop 并发线程数 | open-loop 下不限制最大在途请求数 |
| `--batch-size` | int | `64` | 每个请求的样本数或 item 数 | 正整数 |
| `--duration-secs` | int | `10` | 压测持续秒数 | 正整数 |
| `--target-qps` | int | `0` | open-loop 目标 QPS | `>0` 启用 open-loop；`0` 使用 closed-loop |
| `--input-file` | path | 空 | 使用真实输入文件构造请求 | 不传时生成 synthetic 请求 |
| `--feature-config` | path | 空 | broadcast 真实输入按 feature config 拆 user/item | `--mode broadcast --input-file` 时必填 |
| `--no-header` | flag | false | 真实输入文件无 header | broadcast 输入常用 |
| `--separator` | char | `\t` | broadcast 输入分隔符 | 必须是单字节字符 |

## Rust demo inference

```bash
cargo run --bin demo_inference --release -- \
  examples/shared/feature_config_demo.yaml \
  examples/models/gdcn_esmm.yaml \
  python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  python/artifacts/demo/model_gdcn_esmm_test.csv \
  python/artifacts/demo/model_gdcn_esmm_rust_preds.csv
```

该命令使用位置参数，不使用 `--flag`。

| 位置 | 参数 | 类型 | 说明 | 约束 |
|---:|---|---|---|---|
| 1 | `feature_config.yaml` | path | 特征 DAG 配置 | 必须可读 |
| 2 | `model_config.yaml` | path | 模型配置 | 必须可读 |
| 3 | `model.safetensors` | path | Python 导出的权重 | key/shape 必须匹配模型配置 |
| 4 | `test.csv` | path | 输入 CSV | header 需包含请求特征列 |
| 5 | `output.csv` | path | 输出预测 CSV | 父目录需可写 |

## Rust validate manifest

```bash
cargo run --bin validate_manifest --release -- \
  python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

该命令使用位置参数，不使用 `--flag`。

| 位置 | 参数 | 类型 | 说明 | 约束 |
|---:|---|---|---|---|
| 1..N | `manifest.yaml` | path | 一个或多个 serving manifest | 至少传一个；每个 manifest 的父目录会作为 model dir 加载 |
