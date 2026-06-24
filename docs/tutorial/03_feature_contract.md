# 03. 特征工程契约

[目录](README.md) | [上一章](01_project_structure.md) | [下一章](04_offline_training_flow.md)

推荐排序系统最容易出线上离线不一致的地方是特征工程。scale-rec 的设计原则是：特征配置是一份训练和推理共享的契约，Python 训练和 Rust 推理都按同一份 YAML 执行同一条 DAG。

这一章讲清楚五件事：

1. 原始列如何声明。
2. 在线请求聚合层如何知道字段从哪里取。
3. 原始列如何经过 operators 变成 embeddable feature。
4. embedding、hash、sequence、pooling 如何影响模型输入维度。
5. 改特征后如何保证 Python/Rust 一致。

## 特征配置的三层结构

`examples/shared/feature_config_discover.yaml` 主要由三部分组成：

```yaml
version: 1.0.0
data_sources:
  - ...
sources:
  - ...
operators:
  - ...
```

逻辑上可以分成三层：

| 层 | 配置位置 | 作用 |
|---|---|---|
| 取数来源层 | `data_sources` | 声明在线请求聚合层可以从哪些系统准备原始字段 |
| 原始列层 | `sources` | 声明训练文件和在线请求里可能出现的字段 |
| 预处理层 | `operators` | 把原始字段解析、分桶、交叉、hash、序列化 |
| embedding 层 | operator 的 `embed` | 声明哪些 operator 输出进入模型，以及 embedding 规格 |

当前项目的约定是：`sources` 不直接配置 `embed`，所有入模型的离散特征都通过 operator 输出的 `embed` 声明。这样能保证原始字段一定经过显式预处理后再进入模型。

## data_sources：取数来源契约

`data_sources` 是在线服务契约的一部分，用来告诉请求聚合层字段应该从哪个渠道准备。示例中的 HBase、ES、Flink、Milvus 只是来源类型，实际项目可以按业务命名。

```yaml
data_sources:
  - name: user_profile_hbase
    kind: hbase
    description: user profile and behavior features
  - name: item_document_es
    kind: elasticsearch
    description: item document and metadata features
```

Rust 推理服务不会直接访问这些外部系统。它加载模型导出目录中的 feature config 后，通过 `/models/{model}/features` 暴露这份契约；调用方按契约取数并把字段传给 `/predict` 或 `/predict/broadcast`。

## sources：原始字段契约

source 定义字段名、归属、取数来源、类型、默认值和角色：

```yaml
- name: user_id
  source: User
  data_source: user_profile_hbase
  dtype: int
  default_val: '0'

- name: interest_keywords
  source: User
  data_source: user_profile_hbase
  dtype: string
  default_val: ''

- name: is_click
  dtype: int
  default_val: '0'
  role: label
```

关键字段：

| 字段 | 含义 |
|---|---|
| `name` | 字段名；无 header TSV 时也决定列顺序 |
| `source` | 业务归属，常见为 `Item`、`User`、`Context` |
| `data_source` | 在线取数来源，引用顶层 `data_sources[].name` |
| `dtype` | 原始值解析类型 |
| `default_val` | 缺失或空值时的默认值 |
| `role` | `feature`、`label`、`discard` |

生产接入时，`default_val` 要谨慎选择。比如 `user_id` 的默认值是 `0`，如果业务里 `0` 也是合法用户，就会混淆真实用户和缺失用户。

## operators：DAG 预处理契约

operator 定义一个 DAG 节点：

```yaml
- name: item_id_hash
  op_type: FeatureHash
  inputs:
  - item_id
  outputs:
  - item_id_idx
  params:
    vocab_size: 5000
    num_hashes: 1
  embed:
    vocab_size: 5000
    embed_dim: 16
```

字段含义：

| 字段 | 含义 |
|---|---|
| `name` | DAG 节点名，全局唯一 |
| `op_type` | 算子类型，例如 `FeatureHash`、`Bucketing`、`StringParser` |
| `inputs` | 输入字段，可以是 source，也可以是上游 operator 输出 |
| `outputs` | 输出字段，供下游算子或 embedding 使用 |
| `params` | 算子参数 |
| `embed` | 如果存在，说明这个输出是模型输入特征 |

训练时 `FeatureDag` 会拓扑排序执行 operators。在线推理时 Rust 侧也会解析同一份 operators，因此 Python/Rust 都支持的算子才适合作为线上特征。

如果你要继续查具体算子语义，建议按这几个入口看：

- [特征算子总览](../feature_operators.md#4-全部-17-个算子)
- [算子速查表](../feature_operators.md#7-算子速查表)
- [扩展新算子](../feature_operators.md#8-扩展新算子)

其中最常和本章搭配阅读的是：

- `Bucketing`、`DictMapper`、`FeatureHash`：离散化和低/高基数映射
- `StringParser`、`Split`、`FlatSplit`、`ListStringParser`：字符串和列表解析
- `JsonExtractList`、`SequenceOp`：列表抽取和定长序列化
- `CrossFeature`、`ListOverlap`、`ConcatHash`、`ParsedFeatureHash`：交叉、重叠和融合预处理

## embeddable feature 如何产生

模型不是直接读取所有 source，而是只读取 `FeatureDag.embeddable_features()` 返回的特征。这个列表来自带 `embed` 的 operator 输出。

例如：

```yaml
- name: user_id_hash_a
  op_type: FeatureHash
  inputs:
  - user_id
  outputs:
  - user_id_idx_a
  params:
    vocab_size: 5000
    num_hashes: 1
    namespace: user_id
    salt: a
  embed:
    vocab_size: 5000
    embed_dim: 16
```

这会产生一个 embeddable feature：

```text
user_id_idx_a: vocab_size=5000, embed_dim=16, pooling=first
```

模型里对应的 PyTorch embedding key 是：

```text
embeddings.emb_user_id_idx_a.weight
```

Rust Candle 侧必须用相同路径加载这个权重。

## 标量特征

标量特征通常输出一个整数 bucket，tensor shape 是 `[batch]`：

```yaml
- name: scene_hash
  op_type: FeatureHash
  inputs: [scene]
  outputs: [scene_idx]
  params:
    vocab_size: 10
    num_hashes: 1
  embed:
    vocab_size: 10
    embed_dim: 4
```

处理链路：

```text
scene -> FeatureHash -> scene_idx: int
scene_idx tensor: [batch]
embedding 后: [batch, 4]
```

默认 `pooling` 是 `first`，对标量特征没有额外影响。

## 序列特征

序列特征通常先解析出 list，再逐元素 hash：

```yaml
- name: interest_kw_hash
  op_type: ParsedFeatureHash
  inputs:
  - interest_keywords
  outputs:
  - interest_kw_ids
  params:
    parse_mode: structured
    sep1: '|'
    sep2: '#'
    key_index: 0
    pad_len: 10
    pad_val: ''
    vocab_size: 500
    num_hashes: 1
  embed:
    vocab_size: 500
    embed_dim: 4
    pooling: flatten
```

处理链路：

```text
"人工智能#0.9|新能源#0.7"
  -> ["人工智能", "新能源", "", ...]  # pad 到固定长度
  -> [hash("人工智能"), hash("新能源"), hash(""), ...]
  -> tensor: [batch, seq_len]
  -> embedding: [batch, seq_len, embed_dim]
  -> pooling
```

`seq_len` 可以显式配置，也可以从上游 schema 的固定长度推断。为了让配置更容易审计，生产特征建议显式写出关键序列长度。

## pooling 对模型输入维度的影响

embedding 层支持这些 pooling：

| pooling | 输入 tensor | embedding 后 | 输出到模型 |
|---|---|---|---|
| `first` | `[batch]` 或 `[batch, seq]` | `[batch, dim]` 或 `[batch, seq, dim]` | 标量直接用；序列只取第一个 |
| `mean` | `[batch, seq]` | `[batch, seq, dim]` | `[batch, dim]` |
| `sum` | `[batch, seq]` | `[batch, seq, dim]` | `[batch, dim]` |
| `max` | `[batch, seq]` | `[batch, seq, dim]` | `[batch, dim]` |
| `flatten` | `[batch, seq]` | `[batch, seq, dim]` | `[batch, seq * dim]` |

常见选择：

- ID、枚举、分桶：用默认 `first`。
- 标签序列、兴趣序列：如果要保留每个位置，用 `flatten`；如果只要集合表达，用 `mean/sum/max`。
- `flatten` 会放大输入维度，`seq_len=10`、`embed_dim=16` 会贡献 `160` 维。

## padding 与空序列

序列特征有两类 padding：

1. 解析阶段 padding：例如 `StringParser`、`JsonExtractList`、`ParsedFeatureHash` 的 `pad_len` / `pad_val`。
2. tensor 阶段 padding：`preprocess_batch()` 按 `seq_len` 补 `0`。

这意味着“看起来有值”的 hash bucket 也可能只是 padding token 的 hash。训练侧 feature quality 已经把这类 padding 计入 `feature_quality.emb.<name>.padding_rate`，同时用有效长度计算 `empty_sequence_rate` 和 `mean_length`。

排查序列特征时，不要只看 tensor 里是不是非零，还要看上游解析结果和 padding bucket。

## `num_hashes` 与 hash 空间

`FeatureHash` 的 `num_hashes` 有两种典型用法。

第一种：同一个 embedding 空间内多 hash。

```yaml
params:
  vocab_size: 1000000
  num_hashes: 2
embed:
  vocab_size: 1000000
  embed_dim: 16
  pooling: flatten
  seq_len: 2
```

同一个 `user_id` 会得到两个 bucket，但它们共享同一张 embedding table：

```text
user_id -> [hash(seed=0), hash(seed=1)]
embedding table: embeddings.emb_user_id_idx.weight
输出维度: 2 * embed_dim
```

如果不显式配置 `pooling: mean/sum/max/flatten`，默认 `first` 会只使用第一个 hash，后续 hash 会被丢掉。

第二种：完全独立 embedding 空间。

```yaml
- name: user_id_hash_a
  op_type: FeatureHash
  inputs: [user_id]
  outputs: [user_id_idx_a]
  params:
    vocab_size: 5000
    num_hashes: 1
    namespace: user_id
    salt: a
  embed: {vocab_size: 5000, embed_dim: 16}

- name: user_id_hash_b
  op_type: FeatureHash
  inputs: [user_id]
  outputs: [user_id_idx_b]
  params:
    vocab_size: 5000
    num_hashes: 1
    namespace: user_id
    salt: b
  embed: {vocab_size: 5000, embed_dim: 16}
```

这会生成两张独立 embedding table：

```text
embeddings.emb_user_id_idx_a.weight
embeddings.emb_user_id_idx_b.weight
```

它们读同一个原始 `user_id`，但 hash 前缀不同，bucket 不同，参数也完全独立。代价是模型参数和输入维度都会增加。

## 列表输入上的 `num_hashes`

如果 `FeatureHash` 输入已经是 list，当前实现会按元素逐个 hash，并固定使用单路 hash。也就是说，列表特征配置 `num_hashes > 1` 不会得到 `[list_len, num_hashes]` 这种四维结构。

原因是推荐模型下游通常消费 `[batch, seq_len]` 的离散索引；如果列表元素再叠多 hash，会变成 `[batch, seq_len, num_hashes, dim]`，多数排序模型不能直接消费。

## 特征变更的影响面

不同特征改动影响范围不同：

| 改动 | 是否改变模型输入 | 是否影响旧权重加载 |
|---|---|---|
| 修改 `default_val` | 通常不改变 shape | 权重可加载，但分布会变 |
| 修改 hash `vocab_size` | 改变 embedding shape | 旧权重不能 strict 加载 |
| 修改 `embed_dim` | 改变 embedding shape 和后续层 shape | 旧权重不能 strict 加载 |
| 新增/删除 embeddable feature | 改变输入维度和权重 key | 旧权重不能 strict 加载 |
| 修改 `pooling` / `seq_len` | 可能改变输入维度 | 通常需要重新训练 |
| 修改 `namespace/salt/version` | shape 不变，但 bucket 语义变 | 权重可加载但语义不匹配 |

因此，改特征后不要只看训练能否启动，还要判断旧 safetensors 是否仍然有语义意义。

## 在线离线一致性验证

涉及共享特征逻辑时，至少跑：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  pytest python/tests/test_dag.py python/tests/test_feature_hash.py -v

cargo test --test golden_consistency
```

如果改动会影响模型输入、权重 key 或模型结构，再跑：

```bash
cargo test --test model_smoke

PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

## Debug 检查入口

单样本检查：

```python
from train.core.config import FlowConfig
from train.core.dag import FeatureDag

fc = FlowConfig.from_yaml("examples/shared/feature_config_discover.yaml")
dag = FeatureDag(fc, debug_mode=True)
result = dag.execute({"user_id": 123, "item_id": 456})
print(result.features)
```

最终 tensor 检查：

```python
tensors = dag.preprocess_batch([
    {"user_id": 123, "item_id": 456},
    {"user_id": 124, "item_id": 457},
])
for name, tensor in tensors.items():
    print(name, tuple(tensor.shape), tensor[:2])
```

训练时整体质量看 run manifest 里的：

```text
feature_quality.source.<name>.missing_rate
feature_quality.source.<name>.default_rate
feature_quality.emb.<name>.empty_sequence_rate
feature_quality.emb.<name>.mean_length
feature_quality.emb.<name>.padding_rate
feature_quality.emb.<name>.bucket_utilization
```

更详细的 debug 方法见 [训练手册 - 特征预处理 Debug](../TRAINING_GUIDE.md#特征预处理-debug)。

## 特征接入检查清单

新增或修改特征时，按下面顺序检查：

1. 原始字段是否已在 `sources` 中声明，`dtype/default_val/role` 是否正确。
2. operator 输入是否只依赖 feature 字段，不消费 label 字段。
3. 输出是否需要进入模型；需要才配置 `embed`。
4. `vocab_size/embed_dim/pooling/seq_len` 是否符合业务基数和模型容量。
5. 序列特征的 padding 是否可解释，`padding_rate` 是否可接受。
6. 高基数字段是否使用 `FeatureHash`，低基数枚举是否可以考虑 `DictMapper`。
7. 同一字段多 hash 时，是共享 embedding 空间还是独立 embedding 空间。
8. 改动是否破坏旧权重加载或旧权重语义。
9. Python/Rust 一致性测试是否通过。

下一章会进入离线训练流程，解释这些特征如何被 batch 读取、预处理、送入模型，并如何产生 checkpoint、metric 和发布产物。
