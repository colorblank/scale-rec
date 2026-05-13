# scale-rec

推荐系统特征预处理与模型训练/推理框架。Rust 负责推理引擎，Python 负责训练和权重导出。

## 架构

```
scale-rec/
├── src/                          # Rust 推理引擎 (Candle)
│   ├── feats/
│   │   ├── config.rs             # FlowConfig, DType, SourceDef, OperatorDef
│   │   ├── dag.rs                # FeatureDag: 拓扑排序 + 单样本执行
│   │   ├── metrics.rs            # FeatureMetrics, PerformanceTracer
│   │   └── ops/                  # 7 个特征算子
│   │       ├── bucketing.rs      #   连续值分桶
│   │       ├── cross_feature.rs  #   特征交叉 (内积/笛卡尔积)
│   │       ├── dict_mapper.rs    #   字典映射
│   │       ├── expression.rs     #   Rhai 脚本求值
│   │       ├── plugin.rs         #   cdylib 插件
│   │       ├── sequence.rs       #   序列 pad/truncate
│   │       └── string_parser.rs  #   字符串解析
│   ├── layers/
│   │   ├── embedding.rs          # FeatureEmbeddings
│   │   ├── fm.rs                 # FM 二阶交互
│   │   ├── mlp.rs                # 通用 MLP
│   │   └── towers.rs             # TaskTower, MultiTaskTower, 任务关系推导
│   ├── models/
│   │   ├── mod.rs                # Model trait + ModelConfig 枚举
│   │   ├── lr.rs                 # Logistic Regression
│   │   ├── deepfm.rs             # DeepFM
│   │   ├── mmoe.rs               # MMoE
│   │   ├── esmm.rs               # ESMM
│   │   └── unimixer/             # UniMixer (Tokenizer + UniMixing + SwiGLU + SiameseNorm)
│   ├── lib.rs
│   └── main.rs                   # 集成示例
├── python/                       # Python 训练管线 (PyTorch)
│   └── src/train/                # 镜像 Rust 结构: ops/, layers/, models/
├── examples/
│   └── feature_config.yaml       # 特征预处理 DAG 配置（Rust + Python 共享）
└── python/config/                # 模型配置 YAML
```

## 模型

| 模型 | 论文 | 特点 |
|------|------|------|
| **LR** | — | Embedding + Linear，最简基线 |
| **DeepFM** | Guo et al., 2017 | FM 一阶 + FM 二阶 + Deep MLP |
| **MMoE** | Ma et al., 2018 | 多门控专家混合，每任务独立组合专家 |
| **ESMM** | Ma et al., 2018 | CTR×CVR 乘积链，全量空间消除 SSB |
| **UniMixer** | — | Token 化 + 双随机矩阵交互 + SiameseNorm |

所有模型实现 `Model` trait：`forward(HashMap<name, Tensor>) -> HashMap<task, logits>`。

## 特征预处理

特征流水线由 `examples/feature_config.yaml` 定义，Rust 和 Python 共享同一份配置：

```yaml
version: "1.0.0"
sources:                          # 原始输入，带 embed 配置的自动送入 Embedding
  - {name: user_id, dtype: int, embed: {vocab_size: 10000, embed_dim: 16}}
operators:                        # 算子链 → DAG 拓扑执行
  - {op_type: Bucketing, inputs: [user_age], outputs: [age_bucket], params: {boundaries: [18,25,35,50]}}
  - {op_type: DictMapper, inputs: [category], outputs: [cat_idx], params: {mapping: {elec:1, book:2}}}
```

模型特征规格统一从 `FeatureDag.embeddable_features()` 获取，不在模型配置中重复声明。

## 快速开始

### Rust 推理

```bash
cargo run
# 加载 examples/feature_config.yaml → 预处理 → UniMixer 前向 → 输出 logits
```

### Python 训练 + 导出

```bash
cd python
uv sync
uv run python -m train.main \
  --feature-config ../examples/feature_config.yaml \
  --model-config config/model_deepfm.yaml \
  --data data/train.parquet \
  --epochs 10 --batch-size 128 \
  --export-path model.safetensors
```

### 模型配置

```yaml
# python/config/model_deepfm.yaml
type: deepfm
fm_k: 16
deep_hidden_dims: [256, 128]
```

```yaml
# python/config/model_unimixer.yaml
type: unimixer
token_dim: 16
num_tokens: 2
num_blocks: 2
task_config:
  towers:
    - {name: ctr, hidden_dims: [32], output_dim: 1}
    - {name: cvr, hidden_dims: [32], output_dim: 1}
  relations:
    - {target: ctcvr, sources: [ctr, cvr], op: multiply}
```

### 权重加载到 Rust

Python 训练导出 `model.safetensors`，Rust 端通过 Candle VarMap 直接加载：

```rust
let varmap = VarMap::new();
varmap.load("model.safetensors")?;
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
let model = ModelConfig::DeepFM { fm_k: 16, deep_hidden_dims: vec![256, 128] }
    .build(vb, &features, None)?;
// 权重已自动加载，可直接推理
```

Python 模型参数命名与 Candle `VarBuilder::pp()` 路径完全对齐，无需重命名映射。

## 开发

```bash
# Rust
cargo fmt        # 格式化
cargo check      # 类型检查

# Python
cd python
uvx ruff check src/train/   # Lint
uvx ruff format src/train/  # 格式化
```
