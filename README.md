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
# 启动服务 (manifest 优先；自动加载 python/artifacts/demo 下的 serving manifest)
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64

# 只加载单个模型 manifest
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm.manifest.yaml \
  --worker-threads 4 \
  --blocking-threads 64

# 也可以重复传入多个显式路径；目录路径会扫描其中的 serving manifest
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm.manifest.yaml \
  --model-path python/artifacts/demo/model_discover_unimixer.manifest.yaml

# 兼容旧的松散 .safetensors 产物时，才需要提供 feature-config fallback
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm.safetensors \
  --feature-config examples/feature_config_discover.yaml \
  --worker-threads 4 \
  --blocking-threads 64

# 健康检查
curl http://localhost:8080/health

# 查询已加载模型、默认版本和版本列表
curl http://localhost:8080/models
curl http://localhost:8080/models/model_gdcn_esmm

# Pointwise 推理 (每行一个完整样本)
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'

# 指定版本推理；如果指定版本不可用，可回退到 fallback_version
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","version":"20260526_120000","fallback_version":"20260526_110000","features":[{"user_id":42,"item_id":500}]}'

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
# 启动 Rust HTTP 推理服务；发布 manifest 会指定权重、模型配置和特征配置
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64

# 确认模型和版本已加载
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/models

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

训练会同时产出训练 run 目录和发布产物：

| 路径 | 作用 |
|---|---|
| `python/artifacts/demo/model_gdcn_esmm/20260526_120000/checkpoints/` | 按 epoch/step 保存的历史 checkpoint |
| `python/artifacts/demo/model_gdcn_esmm/20260526_120000/{best,latest}.safetensors` | run 内最佳和最新权重别名 |
| `python/artifacts/demo/model_gdcn_esmm/20260526_120000/run.manifest.yaml` | 训练过程 manifest，不作为 serving 模型加载 |
| `python/artifacts/demo/model_gdcn_esmm.safetensors` | 发布权重，默认复制 best checkpoint |
| `python/artifacts/demo/model_gdcn_esmm.manifest.yaml` | serving manifest，Rust 服务的推荐加载入口 |

serving manifest 记录模型名、版本、模型类型、权重文件、特征配置、模型配置、sha256、任务信息、指标和 `weight_binding`。当前 CLI 没有暴露 `copy_configs` 参数，因此默认会在 manifest 中记录原始 feature/model config 路径及其 sha256；发布目录要长期归档时，需要确保这些配置文件仍然可访问，或在代码配置中启用 `copy_configs` 后再训练。

## 模型加载逻辑

Rust HTTP 服务以 serving manifest 为生产入口。manifest 把权重、模型结构配置和特征配置绑定为一个可校验的模型版本；旧的松散 `.safetensors` 加载只作为兼容模式保留。

### 1. 加载入口

服务支持两类入口：

```bash
# 批量目录入口：扫描目录下所有 serving manifest
target/release/server --model-dir python/artifacts/demo

# 显式路径入口：只加载指定路径，可重复传入
target/release/server \
  --model-path python/artifacts/demo/model_gdcn_esmm.manifest.yaml \
  --model-path python/artifacts/demo/model_discover_unimixer.manifest.yaml
```

入口选择规则：

- 传了 `--model-path`：只加载显式路径，可重复传入；不会再扫描整个 `--model-dir`。
- 未传 `--model-path`：扫描 `--model-dir`。
- 两者都没传：服务拒绝启动。
- 只传 `--model-path`：第一个路径的父目录会作为兼容模式下的 fallback `model_dir`。

### 2. `--model-dir` 批量加载

目录加载流程：

1. 递归扫描 `--model-dir`，最多向下 3 层。
2. 加载匹配以下名称的 serving manifest：
   - `*.manifest.yaml`
   - `*_manifest.yaml`
   - `model_manifest.yaml`
3. 跳过 `run.manifest.yaml`，因为它描述训练过程，不描述 serving 模型。

如果目录中找到了 serving manifest，服务只按 manifest 加载，不再扫描松散 `.safetensors`。如果目录中没有 serving manifest，但传入了 `--feature-config`，服务进入旧兼容模式：扫描目录下 `.safetensors`，用文件名推导模型名和模型配置文件。这种模式只建议用于旧 demo 产物。

### 3. `--model-path` 显式加载

`--model-path` 可以指向三种目标：

| 路径类型 | 行为 |
|---|---|
| serving manifest (`.yaml` / `.yml`) | 按 manifest 加载单个模型版本 |
| 目录 | 扫描该目录中的 serving manifest |
| `.safetensors` | 旧兼容模式，按文件 stem 作为模型名加载，需要 `--feature-config` |

显式 manifest 是推荐方式：

```bash
target/release/server \
  --model-path /models/ranker/20260526_120000/model_manifest.yaml
```

显式旧权重需要 fallback feature config：

```bash
target/release/server \
  --model-path /models/model_gdcn_esmm.safetensors \
  --feature-config examples/feature_config_discover.yaml
```

旧权重模式会用 `.safetensors` 文件 stem 作为模型名，版本固定为 `default`。模型配置 YAML 会按以下位置查找：

1. `.safetensors` 所在目录
2. `--model-dir`
3. `--feature-config` 所在目录

文件名匹配仍沿用 demo 兼容规则，例如 `model_gdcn_esmm.yaml`、`gdcn_esmm.yaml`、去掉 `discover_` 或 `_demo` 后的候选名。

### 4. Serving Manifest

manifest 是模型版本的权威契约。服务会读取：

- `model_id`：HTTP 请求中的模型名
- `model_version`：该模型版本
- `model_type`：模型类型，必须和 model config 的 `type` 一致
- `weights_file`：safetensors 权重文件
- `feature_config_file`：该版本使用的特征配置
- `model_config_file`：该版本使用的模型结构配置
- `weight_binding`：权重命名空间和 strict/extra tensor 校验策略

所有相对路径都以 manifest 所在目录为基准解析。加载前会校验：

- manifest schema version
- feature config sha256
- model config sha256
- weights sha256（如果 manifest 提供）
- model type 是否匹配
- safetensors key 是否缺失、shape 是否匹配、extra tensor 是否允许

`weight_binding` 的默认值与 Python 导出一致：

```yaml
weight_binding:
  format: safetensors
  schema: candle-varbuilder-v1
  root_prefix: ""
  tokenizer_prefix: tokenizer
  unimixer_prefix: unimixer
  strict: true
  allow_extra_tensors: true
```

这些字段控制 Candle `VarBuilder` 的权重命名空间和校验策略。比如 `root_prefix` 非空时，服务会从该 prefix 下加载主模型权重；UniMixer 的主体前缀由 `unimixer_prefix` 控制。

### 5. 多版本注册与选择

同一个 `model_id` 可以加载多个 `model_version`。Registry 内部结构是：

```text
model_id
  ├── version A -> InferenceEngine
  └── version B -> InferenceEngine
```

默认版本按版本字符串取最大值，因此推荐版本号使用可排序格式：

```text
20260526_110000
20260526_120000
20260527_090000
```

查询接口返回已加载模型、默认版本、每个版本的模型类型和 manifest 路径：

```bash
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm
```

`/predict` 和 `/predict/broadcast` 使用同一套版本解析逻辑：

| 请求字段 | 行为 |
|---|---|
| 只传 `model` | 使用该模型默认版本 |
| 传 `model` + `version` | 使用指定版本 |
| 传 `model` + 不存在的 `version` + `fallback_version` | 回退到 fallback 版本 |
| 传 `model` + 不存在的 `version` 且无 fallback | 返回 `REGISTRY_ERROR` |

响应中的 `version` 是实际使用的版本，可能是请求的 `version`，也可能是 `fallback_version`：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "predictions": [
    {"click": 0.73}
  ]
}
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查，返回已加载模型列表 |
| `/models` | GET | 已加载模型、默认版本和版本列表 |
| `/models/{model}` | GET | 指定模型的版本信息 |
| `/predict` | POST | Pointwise: N 行完整特征 → N 个预测 |
| `/predict/broadcast` | POST | Broadcast: 1 user + N items → N 个预测 |

### /models 响应

```json
{
  "models": [
    {
      "name": "model_gdcn_esmm",
      "default_version": "20260526_120000",
      "versions": [
        {
          "version": "20260526_120000",
          "loaded_at": "1780000000",
          "model_type": "gdcn_esmm",
          "manifest_path": "python/artifacts/demo/model_gdcn_esmm.manifest.yaml",
          "is_default": true
        }
      ]
    }
  ]
}
```

### /predict 请求

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "fallback_version": "20260526_110000",
  "features": [
    {"user_id": 42, "item_id": 500}
  ]
}
```

### /predict/broadcast 请求

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "fallback_version": "20260526_110000",
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
  "version": "20260526_120000",
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
cargo fmt && cargo check && cargo test
cargo run --bin server --release -- --model-dir python/artifacts/demo

# Python
uvx ruff check python/src/train/
uvx ruff format python/src/train/
PYTHONPATH=python/src:$PYTHONPATH uv run pytest python/tests/ -v
```
