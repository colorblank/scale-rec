# scale-rec

推荐系统特征预处理与模型训练/推理框架。Rust 负责推理服务和引擎 (Candle)，Python 负责训练和权重导出 (PyTorch)。双端共享同一份特征配置 YAML。

## 架构

```
scale-rec/
├── src/                            # Rust 推理引擎 + HTTP 服务 (Candle)
│   ├── feats/
│   │   ├── config.rs               # FlowConfig, DType, SourceDef, OperatorDef
│   │   ├── dag.rs                   # FeatureDag + ExecutionPlan (预编译 + 列式批量)
│   │   ├── debug/                   # 特征预处理 Debug 追踪器
│   │   └── ops/                    # 9 个特征算子
│   ├── layers/                      # 神经网络层 (Embedding, FM, MLP, Towers)
│   ├── models/                      # discover 主线模型 (GDCN+ESMM / UniMixer)
│   ├── server/                      # HTTP 推理服务
│   │   ├── engine.rs               #   推理引擎 (DAG + Model 封装)
│   │   ├── registry.rs             #   多模型管理 + 热加载
│   │   ├── routes.rs               #   Axum 路由 (/predict, /predict/broadcast 等)
│   │   └── tracing.rs              #   请求级耗时追踪
│   ├── bin/
│   │   ├── server.rs               # 服务启动入口
│   │   ├── bench.rs                # 压测工具 (P50/P95/P99/P99.9/RPS)
│   │   └── demo_inference.rs       # 单次推理示例
│   └── lib.rs
├── python/                          # Python 训练管线 (PyTorch)
│   ├── artifacts/                   #   本地训练产物 (run/best/latest/checkpoints)
│   └── src/train/
│       ├── core/                    #   配置、DAG、任务定义
│       ├── app/                     #   CLI、入口、manifest、artifact 管理
│       ├── training/                #   trainer / loss / metrics / eval / optim
│       ├── debug/                   #   Debug 追踪器
│       ├── ops/                     #   特征算子
│       ├── layers/                  #   神经网络层
│       └── models/                  #   推荐模型 (注册表模式)
├── examples/feature_config_discover.yaml  # 共享特征配置 (discover 示例)
└── Cargo.toml
```

## 特征算子

| 算子 | 用途 | 输入 → 输出 |
|------|------|------------|
| **Bucketing** | 连续值分桶 | f32 → i32 |
| **DictMapper** | 字典映射 | str / list[str] → i32 / list[i32] |
| **StringParser** | 字符串解析 | str → list[str] |
| **ListOverlap** | 列表重叠检测 | list, list → 0/1 |
| **StringConcat** | 字符串拼接 | any.. → str |
| **FeatureHash** | 特征哈希 | any.. → i32 / list[i32] |
| **ExpressionOp** | 表达式求值 | f32.. → f32 |
| **CrossFeature** | 特征交叉 | list, list → f32 / list[str] |
| **SequenceOp** | 序列填充/截断 | list[i32] → list[i32] |
| **PluginOp** | 外部插件 | 动态库加载 |

## 快速开始

### 1. Discover 全流程

```bash
# 生成 discover 合成数据 (2000 行, 38 列)
PYTHONPATH=python/src:$PYTHONPATH uv run python -m scale_rec_demo.generate_discover_data

# 训练 discover 主线模型并导出权重
PYTHONPATH=python/src:$PYTHONPATH uv run python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config examples/model_gdcn_esmm.yaml \
  --epochs 10 --batch-size 128 --no-header \
  --artifact-dir python/artifacts/demo \
  --publish-path python/artifacts/demo/model_gdcn_esmm.safetensors \
  --model-name model_gdcn_esmm

# 端到端验证：训练、导出、PyTorch 推理、Rust 推理、结果比对
PYTHONPATH=python/src:$PYTHONPATH uv run python -m scale_rec_demo.verify_discover_gdcn
```

### 2. HTTP 推理服务

```bash
# 启动服务 (自动加载 python/artifacts/demo 下所有 .safetensors)
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --feature-config examples/feature_config_discover.yaml \
  --worker-threads 4 \
  --blocking-threads 64

# 健康检查
curl http://localhost:8080/health

# Pointwise 推理 (每行一个完整样本)
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'

# Broadcast 推理 (一个用户 + N 个物品)
curl -X POST http://localhost:8080/predict/broadcast -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","user":{"user_id":42},"items":[{"item_id":500},{"item_id":501}]}'
```

### 3. 压测

压测分两类：

- Synthetic 压测：不传 `--input-file`，bench 内部生成通用随机字段。只适合验证 HTTP 链路，不代表 discover 模型真实输入。
- Discover 真实输入压测：传入 discover TSV 和 feature config，bench 会按 `source: User/Context/Item` 构造 `/predict/broadcast` 的 `{user, items}` 请求。性能结论以这个模式为准。

压测前先把服务和 bench 按目标平台重建成对应后端。当前仓库支持的常用组合如下：

| 平台 | 后端 | 构建特征 | 说明 |
|---|---|---|---|
| macOS | Accelerate CPU | `macos-accelerate` | 当前仓库在 macOS 上常用的 CPU 后端 |
| macOS | Metal GPU | `macos-metal` | 需要可用的 Apple GPU，适合看 GPU 推理上限 |
| Linux | MKL CPU | `cpu-mkl` | Linux CPU 压测的推荐后端 |

统一构建命令：

```bash
cargo build --release --features <backend-feature> --bin server --bin bench
```

下面以 demo 发布权重为例。HTTP 请求里的 `model` 是服务实际加载的模型名；先用 `/health` 确认返回列表里包含对应模型，再用同名参数压测：

- `python/artifacts/demo/model_gdcn_esmm.safetensors` → `model_gdcn_esmm`
- `python/artifacts/demo/model_discover_unimixer.safetensors` → `model_discover_unimixer`

```bash
# 启动 Rust HTTP 推理服务
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --feature-config examples/feature_config_discover.yaml \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64

# 确认模型已加载
curl http://127.0.0.1:8080/health

# GDCN+ESMM synthetic smoke，仅验证 HTTP 链路
target/release/bench \
  --target http://127.0.0.1:8080 \
  --model model_gdcn_esmm \
  --mode broadcast \
  --concurrency 10 \
  --batch-size 200 \
  --duration-secs 10 \
  --target-qps 10

# GDCN+ESMM 真实输入压测 (1 user/context + 200 candidates, 300 QPS, 60s)
target/release/bench \
  --target http://127.0.0.1:8080 \
  --model model_gdcn_esmm \
  --mode broadcast \
  --concurrency 300 \
  --batch-size 200 \
  --duration-secs 60 \
  --target-qps 300 \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --no-header

# UniMixer synthetic smoke，仅验证 HTTP 链路
target/release/bench \
  --target http://127.0.0.1:8080 \
  --model model_discover_unimixer \
  --mode broadcast \
  --concurrency 10 \
  --batch-size 200 \
  --duration-secs 10 \
  --target-qps 10

# UniMixer 真实输入压测 (1 user/context + 200 candidates, 300 QPS, 60s)
target/release/bench \
  --target http://127.0.0.1:8080 \
  --model model_discover_unimixer \
  --mode broadcast \
  --concurrency 300 \
  --batch-size 200 \
  --duration-secs 60 \
  --target-qps 300 \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --no-header

# 输出: Scheduled / Target QPS / Success / Errors / RPS / P50 / P95 / P99 / P99.9 / Min/Max
```

300 QPS 验收最低要求：`Scheduled=18000`、`Success=18000`、`Errors=0`、`RPS>=295`。如果压测总耗时明显超过 60 秒，说明服务端已经排队积压。

不同平台的启动方式：

```bash
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --feature-config examples/feature_config_discover.yaml \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

Linux + MKL 时建议补充：

```bash
RUSTFLAGS="-C target-cpu=native" \
cargo build --release --features cpu-mkl --bin server --bin bench

RUST_LOG=warn \
MKL_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
target/release/server \
  --model-dir python/artifacts/demo \
  --feature-config examples/feature_config_discover.yaml \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

macOS + Accelerate 时建议显式构建：

```bash
cargo build --release --features macos-accelerate --bin server --bin bench
```

macOS + Metal 时建议显式构建并启动：

```bash
cargo build --release --features macos-metal --bin server --bin bench

RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --feature-config examples/feature_config_discover.yaml \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

压测时保持 `server` 和 `bench` 使用同一套二进制和同一后端特征，只比较同类结果。不要把 `Accelerate`、`MKL`、`Metal` 的数据直接混在一起看。

### 3. Python 训练 + 导出

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config examples/model_gdcn_esmm.yaml \
  --epochs 10 --batch-size 128 --lr 0.005 --no-header \
  --artifact-dir python/artifacts/demo \
  --publish-path python/artifacts/demo/model_gdcn_esmm.safetensors \
  --model-name model_gdcn_esmm \
  --run-version 20260526_120000
```

训练产物会分成三层：
- `python/artifacts/demo/model_gdcn_esmm/20260526_120000/checkpoints/`：每个 epoch 的 checkpoint
- `python/artifacts/demo/model_gdcn_esmm/20260526_120000/{best,latest}.safetensors`：最佳与最新别名
- `python/artifacts/demo/model_gdcn_esmm.safetensors` 和同目录 `.manifest.yaml`：最终发布权重与 manifest

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回已加载模型列表 |
| `/models` | GET | 已加载模型列表 |
| `/predict` | POST | Pointwise: N 行完整特征 → N 个预测 |
| `/predict/broadcast` | POST | Broadcast: 1 user + N items → N 个预测 |

### /predict 请求

```json
{
  "model": "model_gdcn_esmm",
  "features": [
    {"user_id": 42, "item_id": 500}
  ]
}
```

### /predict/broadcast 请求

```json
{
  "model": "model_gdcn_esmm",
  "user": {"user_id": 42},
  "items": [
    {"item_id": 500},
    {"item_id": 501}
  ]
}
```

### 响应

```json
{
  "model": "model_gdcn_esmm",
  "predictions": [
    {"click": 0.73},
    {"click": 0.21}
  ]
}
```

## 特征配置

特征流水线由 `examples/feature_config_discover.yaml` 定义，Rust/Python 共享同一份 discover schema。

`FeatureDag.embeddable_features()` 提供模型所需的所有特征规格，模型配置中不重复声明。

当前配置支持三类常见类型：
- 标量：`int`、`float`、`string`
- 枚举：`enum`，支持 `values`、`default`、`oov`
- 变长序列：`list`，要求显式 `max_len`/`length`

`FeatureHash`、`DictMapper`、`CrossFeature`、`SequenceOp` 这类算子的输出维度会在 DAG 阶段推导并同步到训练和推理两端。

### 训练、导出与推理验证

发现流式模式的端到端验证脚本会串起训练、权重导出、PyTorch 推理、Rust 推理和输出比对：

```bash
PYTHONPATH=python/src:$PYTHONPATH UV_CACHE_DIR=/private/tmp/uv-cache \
  uv run python -m scale_rec_demo.verify_discover_gdcn
```

如果本地线程环境对 OpenMP 比较敏感，可以先加上：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KMP_INIT_AT_FORK=FALSE
```

## 开发

```bash
# Rust
cargo fmt && cargo check && cargo test   # 24 tests
cargo run --bin server --release          # HTTP 服务

# Python
uvx ruff check src/train/                 # Lint
uvx ruff format src/train/                # 格式化
uv run pytest tests/ -v                   # 14 tests
```
