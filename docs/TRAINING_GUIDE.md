# 训练手册

本文档面向模型训练和发布流程，按“快速跑通、配置说明、训练策略、保存发布、服务加载、压测”的顺序组织。HTTP 请求和响应格式已独立到 [HTTP API](API.md)。

## 阅读顺序

| 章节 | 解决的问题 |
|---|---|
| [快速开始](#快速开始) | 生成 demo 数据、训练 GDCN+ESMM / UniMixer、跑端到端验证 |
| [数据格式](#数据格式) | discover TSV 和训练输入参数 |
| [特征配置](#特征配置) | feature config 的 sources、operators、role |
| [模型配置](#模型配置) | GDCN+ESMM 示例配置 |
| [训练参数](#训练参数) / [训练技巧](#训练技巧) / [评估监控](#评估监控) | 训练超参、优化策略、日志与评估 |
| [保存与推理导出](#保存与推理导出) | checkpoint、发布权重、serving manifest、加载规则 |
| [HTTP 压测](#http-压测) | bench 使用方式和后端构建建议 |

## 快速开始

```bash
# 1. 生成合成数据
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_discover_data

# 2. 训练
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config examples/model_gdcn_esmm.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --publish-path python/artifacts/demo/model_gdcn_esmm.safetensors \
  --model-name model_gdcn_esmm \
  --run-version 20260526_120000

# 2b. 可选：训练 UniMixer
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config examples/model_discover_unimixer.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --publish-path python/artifacts/demo/model_discover_unimixer.safetensors \
  --model-name model_discover_unimixer \
  --run-version 20260526_120000

# 3. 端到端验证
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_discover_gdcn
```

## 数据格式

38 列 Tab 分隔 TSV，无 header。列定义见 `examples/feature_config_discover.yaml`。

**生成合成数据**：`scale_rec_demo.generate_discover_data` 输出 2000 行 × 38 列。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | **必填** | 训练 TSV 路径 |
| `--no-header` | off | 文件无 header 行时启用 |
| `--separator` | `\t` | 字段分隔符 |
| `--null-markers` | NULL \N null None "" | NULL 标记字符串 |
| `--artifact-dir` | `python/artifacts/demo` | 训练 run 目录根路径 |
| `--publish-path` | 自动生成 | 最终发布权重路径 |
| `--model-name` | 自动推导 | 模型逻辑名，用于 run 目录和 manifest |
| `--run-version` | 自动生成 | 训练 run 版本号 |
| `--keep-checkpoints` | 3 | 保留的 checkpoint 数量 |

## 特征配置

`examples/feature_config_discover.yaml` 定义三部分：

- **sources**（38 列）：列名、类型、默认值、角色
- **operators**（68 个）：14 种算子组成的 DAG
- **role 标记**：`feature`（入模型）、`label`（入 loss）、`discard`（读后丢弃）

### role 角色说明

```yaml
sources:
  - name: user_id          # 默认 role=feature
    source: User
    dtype: string
    default_val: ''
  - name: is_click         # 标签列
    dtype: int
    default_val: '0'
    role: label
  - name: answerscore      # 丢弃列（仅问答有效，不进入模型）
    dtype: int
    default_val: '0'
    role: discard
```

## 模型配置

### GDCN+ESMM 门控交叉网络配置

`examples/model_gdcn_esmm.yaml`：

```yaml
type: gdcn_esmm
cross_layers: 3
deep_hidden_dims: [64, 32]
shared_bottom_dims: [32, 16]
label_col_map:
  click: is_click
  cvr: is_cvr
  detail: is_click_detail
  stock: is_click_stock
  stay: stay_time
task_config:
  towers:
    - {name: click, hidden_dims: [16, 8], output_dim: 1, activation: relu}
    - {name: cvr, hidden_dims: [16, 8], output_dim: 1, activation: relu}
    - {name: detail, hidden_dims: [8], output_dim: 1, activation: relu}
    - {name: stock, hidden_dims: [8], output_dim: 1, activation: relu}
    - {name: stay, hidden_dims: [8], output_dim: 1, activation: relu}
  relations:
    - {target: ctcvr, sources: [click, cvr], op: multiply}
    - {target: ctdetail, sources: [click, detail], op: multiply}
    - {target: ctstock, sources: [click, stock], op: multiply}
    - {target: ctstay, sources: [detail, stay], op: multiply}
```

GDCN+ESMM 将门控交叉网络 (GCN) 与 ESMM 多任务预测塔相结合。底层利用 3 层门控交叉层捕捉高阶显式特征交叉，并行使用两层全连接深层网络提取隐式非线性特征；最终通过 5 个独立的预测塔预测单任务 logits，并通过乘积算子 (multiply) 计算多任务联合概率输出。


## 训练参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 30 | 总 epoch 数 |
| `--batch-size` | 64 | 批次大小（pandas chunksize） |
| `--lr` | 0.005 | 初始学习率 |
| `--weight-decay` | 1e-4 | AdamW 权重衰减 |
| `--device` | auto | cpu/cuda/mps/auto |

## 训练技巧

全部默认启用，可通过参数调整或禁用。

### LR Warmup + Cosine Annealing

前 N epoch 线性升温，后续 cosine 降至 `lr × min_lr_ratio`。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--warmup-epochs` | 2 | 预热 epoch 数 |
| `--min-lr-ratio` | 0.01 | 最终 lr 比例 |

### Gradient Clipping

每 batch 裁剪梯度范数，防止爆炸。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--grad-max-norm` | 1.0 | 裁剪阈值（0=禁用） |

### Early Stopping

验证指标连续 N 个 epoch 不提升时自动停止。训练过程中会保存每个 epoch 的 checkpoint，并维护：

- `best.safetensors`：当前最优 checkpoint
- `latest.safetensors`：最新 checkpoint
- `checkpoints/*.safetensors`：按 epoch/step 编号的历史 checkpoint

| 参数 | 默认 | 说明 |
|------|------|------|
| `--early-stopping` | 5 | patience（0=禁用） |

### EMA (Exponential Moving Average)

每个 batch 更新 shadow weights，训练结束后导出 EMA 权重：

```
θ_ema = 0.999 × θ_ema + 0.001 × θ
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--no-ema` | off | 禁用 EMA |
| `--ema-decay` | 0.999 | 衰减率 |

### 不确定性加权损失 (MultiTaskLoss)

Kendall 2018 同方差不确定性，自动平衡：

- **量纲差异**：BCE vs weighted BCE 产生不同 magnitude
- **任务难度**：难任务自动获得更低 σ（更高权重）
- **类别不均衡**：通过 `pos_weight` 差异化少数类惩罚

```
L = Σ exp(-log_var_i) × L_i + 0.5 × log_var_i
```

日志每 epoch 输出：

```
[uncertainty] σ(click)=1.036  σ(cvr)=1.031  σ(detail)=1.006  σ(stay)=1.003  σ(stock)=0.971
```

σ 越低 → 任务学得越好 → 自动分配更高权重。

## 评估监控

### 验证策略

文件头部取 `--eval-samples` 行作为验证集，训练时跳过避免数据泄漏。大数据场景建议预 shuffle 或使用独立验证文件。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--eval-samples` | 2000 | 验证集样本数 |
| `--eval-interval` | 50 | 训练中每隔 N batch 计算 AUC |

### TensorBoard

```bash
# 训练时启用
--tb-dir runs/experiment_name

# 可视化
tensorboard --logdir runs/
```

记录指标：

| 类型 | 指标 | 频率 |
|------|------|------|
| scalar | `train/loss`、`train/lr` | 每 epoch |
| scalar | `val/auc_{task}` | 每 epoch |
| scalar | `grad/pre_clip_norm`、`post_clip_norm` | 每 batch |
| histogram | `grad/{layer}.weight`、`grad/{layer}.bias` | 每 100 batch |

### 日志输出示例

```
device: mps
validation: 512 samples (4 batches)
  batch   10  avg_loss=3.326729  cur_loss=2.968895  lr=2.50e-03
  [timing epoch 1] total=2.7s  batches=12 |
    per_batch: data=17ms(8%) preproc=35ms(17%) forward=19ms(9%)
    loss=46ms(22%) backward=94ms(45%)
epoch   1/10  loss=3.195845  lr=2.50e-03
  click: auc=0.5075  cvr: auc=0.5123  detail: auc=0.4880  stock: auc=0.5317
  [uncertainty] σ(click)=1.015  σ(cvr)=1.015  σ(detail)=1.013
    σ(stay)=1.015  σ(stock)=1.014
checkpoint saved to model.safetensors (best auc=0.5317)
early stopping at epoch 8 (patience=5, best=0.7622@epoch3)
EMA weights exported to model.safetensors
best AUC=0.7622
```

## 保存与推理导出

训练侧由 `TrainingArtifactManager` 管理权重、checkpoint 和 manifest。它的保存逻辑分成 run 产物和 serving 发布产物两层。

### Run 产物

run 目录按 `artifact_root/model_name/run_version` 组织。以上面的快速开始为例，目录是：

```text
python/artifacts/demo/model_gdcn_esmm/20260526_120000/
├── checkpoints/
│   └── epoch-0001-step-000012.safetensors
├── best.safetensors
├── latest.safetensors
└── run.manifest.yaml
```

各文件含义：

| 文件 | 说明 |
|---|---|
| `checkpoints/*.safetensors` | 每次 checkpoint 保存的真实权重文件，文件名包含 epoch 和 step |
| `best.safetensors` | 当前最佳 checkpoint 的别名，由 `publish_best` 控制，默认启用 |
| `latest.safetensors` | 最新 checkpoint 的别名，由 `publish_latest` 控制，默认启用 |
| `run.manifest.yaml` | 训练过程记录，包含 checkpoint 历史、best/latest、发布路径和配置路径 |

`keep_checkpoints` 默认保留 3 个历史 checkpoint，超过后从最旧记录开始删除。`run.manifest.yaml` 是训练记录，不会被 Rust 服务当作 serving manifest 加载。

### 发布产物

发布产物由 `--publish-path` 决定；没有显式传入时，默认是 `artifact_root/model_name.safetensors`。发布权重旁边会生成同 stem 的 serving manifest：

```text
python/artifacts/demo/
├── model_gdcn_esmm.safetensors
└── model_gdcn_esmm.manifest.yaml
```

训练结束时，如果存在 best checkpoint，发布权重默认复制 `best.safetensors`；否则导出当前模型参数。serving manifest 由 `python/src/train/app/manifest.py` 写入，记录：

| 字段 | 说明 |
|---|---|
| `schema_version` | manifest schema，当前为 `1` |
| `model_id` | 服务接口中的模型名，来自 `--model-name` 或自动推导 |
| `model_version` | 服务注册版本，来自 `--run-version` 或自动 UTC 时间戳 |
| `run_version` / `published_version` | 训练 run 版本和发布来源版本 |
| `model_type` | 模型类型，必须和模型配置 YAML 的 `type` 一致 |
| `weights_file` / `weights_sha256` | 发布权重路径和 sha256 |
| `feature_config_file` / `feature_config_sha256` | 特征配置路径和 sha256 |
| `model_config_file` / `model_config_sha256` | 模型配置路径和 sha256 |
| `weight_binding` | safetensors 权重命名、prefix 和校验策略 |
| `tasks` / `label_col_map` / `metrics` | 任务、标签映射和训练指标 |

当前 CLI 没有暴露 `copy_configs` 参数，默认 `copy_configs=false`。因此：

- 如果 feature/model config 和 manifest 在同一发布目录下，manifest 会写相对路径。
- 如果配置文件在仓库 `examples/` 等外部位置，manifest 会写绝对路径。
- 服务加载 manifest 时会按这些路径读取并校验 sha256；长期归档或跨机器部署时，要保证配置文件随发布产物一起可访问，或在代码配置中启用 `copy_configs` 让配置复制到 run 目录。

默认 `weight_binding` 如下：

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

`root_prefix`、`tokenizer_prefix` 和 `unimixer_prefix` 对应 Rust Candle `VarBuilder::pp()` 的权重路径；`strict=true` 时缺少预期 tensor 会加载失败，`allow_extra_tensors=true` 时权重文件里多出的 tensor 会被忽略并记录 warning。

### 服务加载

```bash
target/release/server \
  --model-dir python/artifacts/demo \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

`--model-dir` 会递归扫描最多 3 层目录，加载 `*.manifest.yaml`、`*_manifest.yaml` 和 `model_manifest.yaml`，并跳过 `run.manifest.yaml`。如果扫描到了 serving manifest，服务只按 manifest 加载，不再扫描松散 `.safetensors`。

只加载单个模型版本时，直接指定 serving manifest：

```bash
target/release/server \
  --model-path python/artifacts/demo/model_gdcn_esmm.manifest.yaml \
  --port 8080
```

`--model-path` 可重复传入，也可以指向目录。只要传入了 `--model-path`，服务只加载显式路径，不再扫描整个 `--model-dir`。

| 路径类型 | 行为 |
|---|---|
| `.yaml` / `.yml` | 作为 serving manifest 加载，manifest 是权重、模型配置和特征配置的权威来源 |
| 目录 | 扫描目录内 serving manifest |
| `.safetensors` | 旧兼容模式，按文件名作为模型名加载，需要 `--feature-config` fallback |

只有加载旧的无 manifest `.safetensors` 产物时，才需要提供 `--feature-config` 作为 fallback：

```bash
target/release/server \
  --model-path python/artifacts/demo/model_gdcn_esmm.safetensors \
  --feature-config examples/feature_config_discover.yaml \
  --port 8080
```

旧兼容模式下，服务会把 `.safetensors` 的文件 stem 作为模型名，版本固定为 `default`；模型配置 YAML 会在权重所在目录、`--model-dir` 和 `--feature-config` 所在目录中按候选文件名查找。

manifest 加载时，所有相对路径都基于 manifest 所在目录解析；加载前会校验 feature config、model config、weights 的 sha256、model type，以及 safetensors key/shape。相同 `model_id` 可以加载多个 `model_version`，默认版本按版本字符串取最大值。因此建议 `--run-version` 使用可排序时间戳，例如 `20260526_120000`。

查询模型、指定版本调用和 fallback 的接口格式见 [HTTP API](API.md)。

## HTTP 压测

压测 discover 模型时不要只用 bench 默认随机数据。默认随机数据是通用 synthetic schema，只适合验证 HTTP 链路；真实性能压测必须使用 discover TSV + feature config，让 bench 按 `User/Context/Item` 拆分并构造 `/predict/broadcast` 请求。

GDCN+ESMM 和 UniMixer 实测报告见 `docs/http_benchmark_report.md`。

压测前先按目标平台重建 server 和 bench，并保持两者使用同一套后端特征。常用组合如下：

| 平台 | 后端 | 构建特征 | 说明 |
|---|---|---|---|
| macOS | Accelerate CPU | `macos-accelerate` | macOS 上的默认 CPU 压测选择 |
| macOS | Metal GPU | `macos-metal` | 适合验证 GPU 推理上限 |
| Linux | MKL CPU | `cpu-mkl` | Linux CPU 压测的推荐后端 |

下面以快速开始生成的发布权重为例。HTTP 请求里的 `model` 是服务实际加载的模型名；先用 `/health` 确认返回列表里包含对应模型，再用同名参数压测：

- `python/artifacts/demo/model_gdcn_esmm.safetensors` → `model_gdcn_esmm`
- `python/artifacts/demo/model_discover_unimixer.safetensors` → `model_discover_unimixer`

```bash
# 启动 Rust HTTP 推理服务
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64

# 确认模型和版本已加载；如果执行了 UniMixer 训练，应同时看到 model_discover_unimixer
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

# GDCN+ESMM 真实 discover 输入压测: 1 user/context + 200 candidates, 300 QPS, 60s
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

# UniMixer 真实 discover 输入压测: 1 user/context + 200 candidates, 300 QPS, 60s
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
```

验收最低要求：`Scheduled=18000`、`Success=18000`、`Errors=0`、`RPS>=295`。如果压测进程在 60 秒后仍长时间等待未完成请求，说明服务端已产生排队积压。

平台/后端构建方式：

```bash
# macOS + Accelerate
cargo build --release --features macos-accelerate --bin server --bin bench

# macOS + Metal
cargo build --release --features macos-metal --bin server --bin bench

# Linux + MKL
RUSTFLAGS="-C target-cpu=native" \
cargo build --release --features cpu-mkl --bin server --bin bench
```

平台/后端启动方式：

```bash
# 通用启动参数，适用于 macOS Accelerate / macOS Metal / Linux MKL
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

Linux + MKL 时建议设置：

```bash
RUST_LOG=warn \
MKL_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
target/release/server \
  --model-dir python/artifacts/demo \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

压测时只比较同一平台、同一后端、同一构建参数下的结果。不要把 `Accelerate`、`Metal`、`MKL` 的结果混用。

## 代码架构

```
python/src/train/
├── core/        — FlowConfig、FeatureDag、TaskSpec、schema
├── app/         — CLI、入口、artifact/manifest 管理
├── training/    — trainer、loss、metrics、eval、optim、quality
├── models/      — discover 主线模型 (GDCN+ESMM / UniMixer)
├── layers/      — MLP、Embedding、Tokenizer、Towers
└── ops/         — 特征算子
python/src/scale_rec_demo/
├── generate_discover_data.py  — 合成数据生成
├── verify_discover_gdcn.py    — discover 主线一致性验证
└── paths.py                   — demo 路径常量
python/artifacts/demo/         — 本地训练输出
```
