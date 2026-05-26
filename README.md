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
│   ├── models/                      # 6 个推荐模型 (LR/DeepFM/MMoE/ESMM/UniMixer/GDCNESMM)
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
│   ├── configs/                     #   Python 模型和 demo 配置 YAML
│   ├── artifacts/                   #   本地生成产物 (safetensors, CSV, debug)
│   ├── data/                        #   训练数据 (parquet)
│   └── src/train/
│       ├── config.py                #   FlowConfig (镜像 Rust config.rs)
│       ├── dag.py                   #   FeatureDag (execute_batch 列式批量)
│       ├── debug/                   #   Debug 追踪器
│       ├── ops/                     #   9 个特征算子
│       ├── layers/                  #   神经网络层
│       ├── models/                  #   6 个推荐模型 (注册表模式)
│       ├── export.py                #   safetensors 导出
│       └── main.py                  #   训练入口
├── examples/feature_config.yaml     # 共享特征配置 (完整示例: 82 特征, 85 算子)
└── Cargo.toml
```

## 模型

| 模型 | 类型 | 输出 | 特点 |
|------|------|------|------|
| **LR** | 单任务 | `pred` | Embedding + Linear，最简基线 |
| **DeepFM** | 单任务 | `pred` | FM 一阶 + 二阶交互 + Deep MLP |
| **MMoE** | 多任务 | 自定义 (ctr, cvr) | 多门控专家混合 |
| **ESMM** | 多任务 | ctr, cvr, ctcvr | CTR×CVR 乘积链，全量空间消除 SSB |
| **UniMixer** | 多任务 | 自定义 (ctr, cvr, ctcvr) | Token 化 + 双随机矩阵交互 + SiameseNorm |
| **GDCN+ESMM** | 多任务 | click, cvr, detail, stock, stay 及乘积关系 | 门控交叉网络 (GCN) + 共享表示层 + 5 任务预测塔 |

新增模型无需修改现有文件：使用 Python `@register_model` + Rust `REGISTRY` 注册即可。

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

### 1. Demo 全流程

```bash
cd python

# 生成合成数据 (2000 行, ctr+cvr 标签)
uv run python -m scale_rec_demo.generate_data

# 训练所有模型 (5 epochs)
uv run python -m scale_rec_demo.train_all --epochs 5

# 验证 PyTorch vs Rust 推理一致性
uv run python -m scale_rec_demo.verify_all

# 验证 GDCN+ESMM 预处理与模型输出一致性 (发现流式模式数据)
uv run python -m scale_rec_demo.verify_discover_gdcn

# Debug 追踪 (逐算子 I/O)
uv run python -m scale_rec_demo.train_all --epochs 1 --models lr --debug-trace 10
```

### 2. HTTP 推理服务

```bash
# 启动服务 (自动加载 temp/ 下所有 .safetensors)
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --feature-config python/configs/demo/legacy/feature_config.yaml

# 健康检查
curl http://localhost:8080/health

# Pointwise 推理 (每行一个完整样本)
curl -X POST http://localhost:8080/predict -H 'Content-Type: application/json' \
  -d '{"model":"model_lr","features":[{"user_id":42,"item_id":500}]}'

# Broadcast 推理 (一个用户 + N 个物品)
curl -X POST http://localhost:8080/predict/broadcast -H 'Content-Type: application/json' \
  -d '{"model":"model_lr","user":{"user_id":42},"items":[{"item_id":500},{"item_id":501}]}'
```

### 3. 压测

```bash
# 生产级压测 (batch=200, broadcast, 60s)
cargo run --bin bench --release -- \
  --target http://localhost:8080 \
  --model model_lr --mode broadcast \
  --concurrency 10 --batch-size 200 --duration-secs 60

# 输出: Total / Errors / RPS / P50 / P95 / P99 / P99.9 / Min/Max
```

### 4. Python 训练 + 导出

```bash
cd python
uv run python -m train.main \
  --feature-config ../examples/feature_config.yaml \
  --model-config configs/models/model_lr.yaml \
  --data data/train.parquet \
  --epochs 10 --batch-size 64 --lr 0.001 \
  --export-path model.safetensors
```

## 性能 (Release build)

### 优化历程 (LR, batch=64, concur=10)

| 阶段 | 技术 | P50 | P99 | 提速 |
|------|------|-----|-----|------|
| 串行逐行 | execute() per row | 239ms | 620ms | 1× |
| 列式批量 | execute_batch + process_batch | 75ms | 192ms | 3.2× |
| Fv 枚举 | 消除 `Arc<dyn Any>` vtable | 73ms | 181ms | 3.4× |
| 广播预计算 | 用户特征一次处理 + skip_ops | 43ms | 93ms | 6.7× |
| 预编译计划 | 零 HashMap 查找 (整数列索引) | 10.9ms | 18.9ms | 33× |

### 生产级 (batch=200, broadcast, 60s, 0 errors)

| 模型 | RPS | P50 | P99 | P99.9 |
|------|-----|-----|-----|-------|
| **LR** | 349 | 20.5ms | 30.5ms | 48.0ms |
| **DeepFM** | 323 | 22.7ms | 34.6ms | 55.5ms |
| **MMoE** | 356 | 20.4ms | 28.5ms | 34.9ms |
| **ESMM** | 356 | 20.4ms | 28.7ms | 37.3ms |
| **UniMixer** | 345 | 21.3ms | 29.9ms | 37.8ms |

**全部模型 P99 < 35ms，满足 300 QPS × 200 候选物品生产要求。**

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
  "model": "model_lr",
  "features": [
    {"user_id": 42, "user_age": 28.5, "item_id": 500, "item_category": "electronics"}
  ]
}
```

### /predict/broadcast 请求

```json
{
  "model": "model_lr",
  "user": {"user_id": 42, "user_age": 28.5},
  "items": [
    {"item_id": 500, "item_category": "electronics"},
    {"item_id": 501, "item_category": "books"}
  ]
}
```

### 响应

```json
{
  "model": "model_lr",
  "predictions": [
    {"pred": 0.73},
    {"pred": 0.21}
  ]
}
```

## 特征配置

特征流水线由 YAML 定义，Rust/Python 共享。示例包含 72 个可嵌入特征和 85 个算子：

```yaml
version: "1.0.0"
sources:
  - name: user_id
    dtype: int
    embed: {vocab_size: 500, embed_dim: 16}
  - name: user_age
    dtype: float
operators:
  - name: age_bucket_op
    op_type: Bucketing
    inputs: [user_age]
    outputs: [user_age_bucket]
    params: {boundaries: [18, 25, 35, 45, 55]}
    embed: {vocab_size: 6, embed_dim: 4}
```

`FeatureDag.embeddable_features()` 提供模型所需的所有特征规格，模型配置中不重复声明。

## 开发

```bash
# Rust
cargo fmt && cargo check && cargo test   # 24 tests
cargo run --bin server --release          # HTTP 服务

# Python
cd python
uvx ruff check src/train/                 # Lint
uvx ruff format src/train/                # 格式化
uv run pytest tests/ -v                   # 14 tests
```
