# 训练手册

## 快速开始

```bash
# 1. 生成合成数据
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/generate_discover_data.py

# 2. 训练
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/train_discover.py \
  --data python/demo/temp/discover_train_data.txt \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400
```

## 数据格式

43 列 Tab 分隔 TSV，无 header。列定义见 `examples/feature_config_discover.yaml`。

**生成合成数据**：`python/demo/generate_discover_data.py` 输出 2000 行 × 43 列。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | **必填** | 训练 TSV 路径 |
| `--no-header` | off | 文件无 header 行时启用 |
| `--separator` | `\t` | 字段分隔符 |
| `--null-markers` | NULL \N null None "" | NULL 标记字符串 |

## 特征配置

`examples/feature_config_discover.yaml` 定义三部分：

- **sources**（43 列）：列名、类型、默认值、角色
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

`python/demo/configs/discover/model_esmm.yaml`：

```yaml
type: esmm
shared_bottom_dims: [32, 16]
click_hidden_dims: [16, 8]
cvr_hidden_dims: [16, 8]
detail_hidden_dims: [8]
stock_hidden_dims: [8]
stay_hidden_dims: [8]
```

5 任务 ESMM 概率关系：
- P(detail) = σ(click) × σ(detail)
- P(stock) = σ(click) × σ(stock)
- P(cvr) = σ(click) × σ(cvr)
- P(stay) = σ(detail) × σ(stay)

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

验证 AUC 连续 N epoch 不提升时自动停止。仅最佳 AUC 时保存 checkpoint。

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

## 推理导出

训练输出 `model.safetensors`（72 个张量），可直接由 Rust Candle 引擎加载：

```bash
cargo run --bin server -- --feature-config examples/feature_config_discover.yaml
```

## 代码架构

```
python/src/train/
├── config.py      — FlowConfig, SourceDef, Role 定义
├── dag.py         — FeatureDag 执行引擎
├── data.py        — stream_file_batches 流式读取
├── metrics.py     — MultiTaskLoss, AUC, _sigmoid
├── trainer.py     — Trainer + TrainConfig + 训练技巧
├── export.py      — safetensors 导出
├── models/        — ESMM, MMoE, DeepFM, LR, UniMixer
├── layers/        — MLP, Embedding, Tokenizer, Towers
└── ops/           — 14 种特征算子
python/demo/
├── train_discover.py          — CLI 入口
├── generate_discover_data.py  — 合成数据生成
└── TRAINING_GUIDE.md          — 本手册
```
