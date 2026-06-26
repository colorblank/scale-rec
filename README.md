# scale-rec

scale-rec 是一个推荐系统训练与推理框架：Python 侧负责样本读取、特征预处理、PyTorch 训练和 safetensors 权重导出；Rust 侧基于 Candle 加载同一份特征配置、模型配置和权重，提供 HTTP 推理服务。

核心约定：

- 特征配置由 YAML 统一描述，Python 训练和 Rust 推理共享。
- 模型结构由 model config YAML 描述，训练权重导出为 safetensors。
- 生产加载推荐使用 serving manifest，把权重、模型配置、特征配置、版本和校验信息绑定在一起。

## Documentation

| 入口 | 内容 |
|---|---|
| [Docs Home](docs/index.md) | 文档首页，按 Get Started / Tutorials / How-to / Reference / Notes 组织 |
| [Getting Started](docs/getting_started.md) | 生成 demo 数据、训练、导出、启动服务、端到端验证 |
| [Tutorials](docs/tutorials/index.md) | 按推荐系统链路学习：样本、特征、训练、发布、推理和排障 |
| [How-to Guides](docs/how_to/index.md) | 面向具体任务的操作指南，例如独立验证集、多日训练、服务加载和性能调优 |
| [Reference](docs/reference/index.md) | CLI、配置、HTTP API、Prometheus 指标、特征算子和 artifact 格式 |
| [Notes](docs/notes/index.md) | 架构、性能、压测报告和设计改进记录 |
| [Development](docs/reference/development.md) | 本地开发、测试、格式化和端到端验证命令 |

## Quickstart

所有命令从仓库根目录执行。

```bash
# 1. 生成 demo 数据
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_demo_data \
  --label-policy examples/shared/demo_label_policy.yaml

# 2. 训练并导出一个 GDCN+ESMM 模型
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm \
  --run-version 20260526_120000

# 3. 端到端验证 Python 训练导出与 Rust 推理一致性
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all \
  --models demo_lr,demo_gdcn_esmm,demo_unimixer,demo_token_mixer_large,demo_rankmixer \
  --force-train
```

更多可运行命令见 [Getting Started](docs/getting_started.md)。

## Serving

训练导出的标准 serving 目录包含：

```text
serving/
├── model.manifest.yaml
├── model.safetensors
└── configs/
    ├── feature_config.yaml
    └── model_config.yaml
```

推荐按 manifest 或 model directory 加载：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

只加载单个模型版本：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

HTTP 请求格式见 [HTTP API Reference](docs/reference/http_api.md)；模型加载规则见 [Getting Started](docs/getting_started.md#serve-the-model) 和 [Rust Model Loading](docs/reference/rust_model_loading.md)。

## Repository layout

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
├── examples/                       # demo 示例共享配置和模型配置
├── docs/                           # 文档
├── docker/                         # Docker 打包入口
├── tests/                          # Rust 集成测试
└── Cargo.toml
```

## Validation

常用开发验证命令：

```bash
cargo fmt
cargo check
cargo test

uvx --offline ruff check python/src/
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/ -q
```

涉及特征、模型、权重命名或推理一致性的改动，跑端到端验证：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `scale_rec_demo.generate_demo_data` | `--label-policy` selects demo label policy YAML | [CLI Reference: Generate demo data](docs/reference/cli.md#generate-demo-data) |
| `train.app.main demo` | `--data` / `--feature-config` / `--model-config` / `--train-config` / `--epochs` / `--batch-size` / `--no-header` / `--eval-samples` / `--artifact-dir` / `--model-name` / `--run-version` | [CLI Reference: Train demo](docs/reference/cli.md#train-demo) |
| `scale_rec_demo.verify_all` | `--models` selects model keys; `--force-train` retrains before comparison | [CLI Reference: Verify all](docs/reference/cli.md#verify-all) |
| `cargo run --bin server` | `--model-dir` scans a serving directory; `--model-path` loads one manifest | [CLI Reference: Rust server](docs/reference/cli.md#rust-server) |
| `cargo fmt` / `cargo check` / `cargo test` | No project-specific flags in this page | [Development Reference](docs/reference/development.md) |
| `uvx --offline ruff check` | `--offline` avoids network access; `check` runs linting | [Development Reference](docs/reference/development.md) |
| `pytest` | `python/tests/ -q` runs Python tests quietly | [Development Reference](docs/reference/development.md) |
