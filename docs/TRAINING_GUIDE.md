# 训练手册

本文档面向模型训练和发布流程，按“快速跑通、配置说明、训练策略、保存发布、服务加载、压测”的顺序组织。HTTP 请求和响应格式已独立到 [HTTP API](API.md)。

## 阅读顺序

| 章节 | 解决的问题 |
|---|---|
| [快速开始](#快速开始) | 生成 demo 数据、训练 GDCN+ESMM / UniMixer、跑端到端验证 |
| [数据格式](#数据格式) | discover TSV 和训练输入参数 |
| [特征配置](#特征配置) | feature config 的 sources、operators、role |
| [训练流程](#训练流程) | 数据、标签、模型、训练配置如何组合 |
| [模型配置](#模型配置) | GDCN+ESMM / UniMixer 的任务级配置 |
| [训练参数](#训练参数) / [训练技巧](#训练技巧) / [评估监控](#评估监控) | 训练超参、优化策略、日志与评估 |
| [保存与推理导出](#保存与推理导出) | checkpoint、发布权重、serving manifest、加载规则 |
| [HTTP 压测](#http-压测) | bench 使用方式和后端构建建议 |

## 快速开始

```bash
# 1. 生成合成数据
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_discover_data \
  --label-policy examples/discover_label_policy.yaml

# 2. 训练
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config examples/model_gdcn_esmm.yaml \
  --train-config examples/train_defaults.yaml \
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
  --train-config examples/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --publish-path python/artifacts/demo/model_discover_unimixer.safetensors \
  --model-name model_discover_unimixer \
  --run-version 20260526_120000

# 3. 端到端验证
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

## 训练流程

训练链路分成 4 层配置，默认优先级从低到高是：`train_defaults.yaml` < 模型 YAML < 命令行参数。

1. `examples/feature_config_discover.yaml`
   定义原始输入列、特征算子 DAG、每个列的 `role`。它决定哪些列进入模型，哪些列作为标签，哪些列只是中间产物。
2. `examples/discover_label_policy.yaml`
   定义 demo 数据生成时的标签规则。它只影响合成数据，不参与模型前向。
3. `examples/train_defaults.yaml`
   定义训练默认值，包括 batch size、optimizer、eval 样本数、warmup、early stopping、EMA、TensorBoard 等。CLI 可以覆盖其中任意项。
4. `examples/model_*.yaml`
   定义模型结构和任务语义。`tasks:` 是训练和评估的单一事实来源，决定每个任务用哪个 label、什么 loss、统计哪些 metrics。

典型执行顺序如下：

1. 先生成 demo 数据，写出 TSV 和标签列。
2. 再加载 feature config，把原始列编排成模型输入。
3. 再读取 model config，解析 `tasks:` 和 `task_config:`。
4. 再读取 train config，合并 CLI 覆盖项。
5. 训练时按 task 计算 loss，评估时按 task 计算 metrics。
6. 最终导出 safetensors 权重和 manifest。

## 数据格式

45 列 Tab 分隔 TSV，无 header。列定义见 `examples/feature_config_discover.yaml`。

**生成合成数据**：`scale_rec_demo.generate_discover_data` 输出 2000 行 × 45 列，其中 38 列是特征输入，7 列是监督标签。

标签列包含 `is_click`、`is_cvr`、`is_click_detail`、`is_click_stock`、`stay_time_label` 等字段；具体是否启用某个派生标签，由 `examples/discover_label_policy.yaml` 控制。

如果你要同时训练 GDCN+ESMM 和 UniMixer，建议保留完整标签集合，复用同一份 TSV。

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
| `--train-config` | `examples/train_defaults.yaml` | 训练超参、优化器、评估默认值 |
| `--label-policy` | `examples/discover_label_policy.yaml` | 仅用于合成数据生成的标签规则 |

## 特征配置

`examples/feature_config_discover.yaml` 定义三部分：

- **sources**（45 列）：列名、类型、默认值、角色
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
  stay: stay_time_label
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

`tasks:` 是训练配置的核心，字段含义如下：

| 字段 | 说明 |
|---|---|
| `name` | 任务名，同时也是日志、metric、tower 的主键 |
| `label` | 该任务对应的标签列 |
| `loss` | 该任务使用的 loss 名称，必须由训练代码注册 |
| `metrics` | 该任务评估时统计的指标列表 |

`label_col_map` 负责把任务名映射到真实列名，训练、导出和 manifest 都会使用它。`task_config` 负责定义 tower 和关系结构，属于模型前向的一部分。两者职责不同，不要混用。

GDCN+ESMM 将门控交叉网络与 ESMM 多任务预测塔相结合。底层利用 3 层门控交叉层捕捉高阶显式特征交叉，并行使用两层全连接深层网络提取隐式非线性特征；最终通过 5 个独立的预测塔输出任务 logits，并通过乘积关系计算联合概率。

### UniMixer 配置

`examples/model_discover_unimixer.yaml` 复用同一套 `tasks:` 约定，区别在于 `type: unimixer`，以及 UniMixer 自身的 token、block、rank 等结构参数。训练、评估和导出层面对它的处理方式与 GDCN+ESMM 一致。

### 任务定义建议

- 分类任务常用 `loss: bce`，`metrics: [auc, logloss]`
- 回归任务常用 `loss: mse` 或业务自定义回归 loss，`metrics: [mae, mse]`
- 如果某个任务没有可用标签，就不要放进 `tasks:`，不要依赖代码里的默认兜底

模型配置是训练流程里的第一手任务定义。后续的 trainer、evaluator、manifest 都只读这里，不再根据模型类型猜测任务。


## 训练参数

下面这些参数来自 `examples/train_defaults.yaml`，CLI 只负责覆盖，不再在代码里写死。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--epochs` | 30 | 总 epoch 数 |
| `--batch-size` | 64 | 批次大小（pandas chunksize） |
| `--lr` | 0.005 | 初始学习率 |
| `--weight-decay` | 1e-4 | AdamW 权重衰减 |
| `--device` | auto | cpu/cuda/mps/auto |
| `--eval-samples` | 400 | 从文件头部切出的验证样本数 |
| `--eval-interval` | 50 | 训练中每隔 N batch 触发一次评估 |
| `--log-interval` | 10 | 训练中每隔 N batch 打一次日志 |
| `--warmup-steps` | 200 | 学习率 warmup 步数 |
| `--min-lr-ratio` | 0.01 | Cosine annealing 最低学习率比例 |
| `--grad-max-norm` | 1.0 | 梯度裁剪阈值 |
| `--early-stopping-patience` | 5 | 验证指标连续多少次不提升后停止 |
| `--ema-decay` | 0.999 | EMA 衰减率 |
| `--loss-weighting` | static | 多任务 loss 的加权策略 |

### 训练默认配置

`examples/train_defaults.yaml` 是训练超参的基线配置。它控制训练行为，但不定义模型结构：

```yaml
epochs: 30
batch_size: 64
optim:
  name: adamw
  lr: 0.005
  weight_decay: 0.0001
  emb_lr: null
  emb_weight_decay: null
eval_samples: 400
eval_interval: 50
log_interval: 10
lr_schedule:
  warmup_steps: 200
  min_lr_ratio: 0.01
grad_max_norm: 1.0
early_stopping_patience: 5
ema_decay: 0.999
loss_weighting: static
tb_dir: ""
eval:
  metrics: [auc]
  monitor_metric: auc
  log_path: ""
  gauc_group_feature: user_id
```

字段含义：

- `epochs` 和 `batch_size`：控制训练轮数和每次喂入模型的 batch 大小
- `optim`：优化器类型和学习率参数，`emb_lr`、`emb_weight_decay` 允许把 embedding 参数和其它参数分开优化
- `eval_samples`：从文件头部切出的验证样本数
- `eval_interval`：训练中每隔多少个 batch 做一次评估
- `log_interval`：训练中每隔多少个 batch 打一次进度日志
- `lr_schedule.warmup_steps`：学习率线性 warmup 的步数
- `lr_schedule.min_lr_ratio`：cosine 衰减的最低学习率比例
- `grad_max_norm`：梯度裁剪阈值
- `early_stopping_patience`：验证指标连续多少次不提升后停止
- `ema_decay`：EMA 影子权重的更新衰减率
- `loss_weighting`：多任务 loss 的加权策略，当前支持 `equal`、`static`、`uncertainty`
- `tb_dir`：TensorBoard 输出目录，空字符串表示不写
- `eval.metrics`：默认验证指标列表。对于已经在 `tasks:` 里声明 metrics 的模型，这里主要作为兜底和兼容配置
- `eval.monitor_metric`：early stopping 监控的主指标。训练器会在所有任务里取这个 metric 的最好值作为监控分数
- `eval.log_path`：额外保存评估日志的路径
- `eval.gauc_group_feature`：GAUC 分组特征名

### 标签策略配置

`examples/discover_label_policy.yaml` 只参与 demo 数据生成，不影响模型结构和推理。它的作用是把原始字段转成监督标签：

```yaml
version: 1.0.0
click:
  quality_weight: 0.45
  item_type_bonus:
    news: 0.12
    report: 0.12
  source_name_bonus:
    同花顺: 0.10
    东方财富: 0.10
  scene_max: 3
  scene_bonus: 0.08
  new_user_label: 新用户
  new_user_bonus: 0.06
  stay_time_threshold: 180
  stay_time_bonus: 0.05
  threshold: 0.42
detail:
  quality_threshold: 0.58
  item_types: [news, report]
stock:
  min_stock_count: 3
  source_name: 雪球
cvr:
  quality_threshold: 0.68
  stay_time_threshold: 240
stay_time_label:
  click_multiplier: [0.85, 0.15]
  noise_min: -25
  noise_max: 45
```

字段含义：

- `click`：点击标签的打分规则，综合质量、来源、场景、新用户偏置等因素
- `detail`：细分点击标签，通常只对特定 `item_types` 生效
- `stock`：股票类点击标签，依赖最小持仓数和来源名
- `cvr`：转化标签，结合质量阈值和停留时间阈值
- `stay_time_label`：停留时长标签，基于点击信号叠加噪声生成

## 训练技巧

这些策略的具体数值默认来自 `examples/train_defaults.yaml`。是否启用、采用什么阈值，都应从配置读取。

### 学习率调度

前 `warmup_steps` 个 step 线性升温，之后使用 cosine 衰减到 `lr × min_lr_ratio`。这里按 step 而不是按 epoch 计算，避免不同 batch size 下调度节奏漂移。

### Gradient Clipping

每 batch 裁剪梯度范数，防止爆炸。

### Early Stopping

验证指标连续 `early_stopping_patience` 次评估不提升时自动停止。训练过程中会保存每个 checkpoint，并维护：

- `best.safetensors`：当前最优 checkpoint
- `latest.safetensors`：最新 checkpoint
- `checkpoints/*.safetensors`：按 epoch/step 编号的历史 checkpoint

### EMA (Exponential Moving Average)

每个 batch 更新 shadow weights，训练结束后导出 EMA 权重：

```
θ_ema = 0.999 × θ_ema + 0.001 × θ
```

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

文件头部取 `eval_samples` 行作为验证集，训练时跳过避免数据泄漏。大数据场景建议预 shuffle 或使用独立验证文件。`eval_interval` 控制的是触发评估的频率，不限定评估指标。

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
| scalar | `val/{task}_{metric}` | 每次评估 |
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
  click: auc=0.5075  logloss=0.6812  cvr: auc=0.5123  logloss=0.6734
  detail: auc=0.4880  logloss=0.6911  stock: auc=0.5317  logloss=0.6658
  stay: mae=285.8082  mse=148712.9375
  [uncertainty] σ(click)=1.015  σ(cvr)=1.015  σ(detail)=1.013
    σ(stay)=1.015  σ(stock)=1.014
checkpoint saved to model.safetensors (best val/click_auc=0.5317)
early stopping at epoch 8 (patience=5, best=val/click_auc=0.7622@epoch3)
EMA weights exported to model.safetensors
best metric=val/click_auc=0.7622
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
├── verify_all.py              — discover 主线与 UniMixer 一致性验证
└── paths.py                   — demo 路径常量
python/artifacts/demo/         — 本地训练输出
```
