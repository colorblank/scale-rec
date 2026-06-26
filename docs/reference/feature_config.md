# Feature Config Reference

feature config YAML 是 Python 训练和 Rust 推理共享的特征契约。它定义原始输入列、在线字段来源、算子 DAG、embedding 配置和 label/discard 角色。

完整算子参考见 [Feature operators](../feature_operators.md)。

## Top-level schema

```yaml
version: "1.0.0"

data_sources:
  - name: request
    kind: request
    description: fields directly provided by the request

sources:
  - name: user_id
    source: User
    data_source: request
    dtype: int
    default_val: "0"
    role: feature

operators:
  - name: user_hash
    op_type: FeatureHash
    inputs: [user_id]
    outputs: [user_id_idx]
    params:
      vocab_size: 100000
      num_hashes: 1
    embed:
      vocab_size: 100000
      embed_dim: 16
```

## `data_sources`

`data_sources` 描述在线请求聚合层应该从哪些系统准备字段。它不改变 DAG 执行逻辑；训练和推理仍只消费已经传入样本或请求的字段。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `name` | string | 是 | 来源名，被 `sources[].data_source` 引用 |
| `kind` | string | 是 | 来源类型，例如 `request`、`hbase`、`elasticsearch`、`flink`、`milvus` |
| `description` | string | 否 | 人类可读说明 |

## `sources`

`sources` 定义原始输入列。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `name` | string | 是 | 原始字段名 |
| `dtype` | string | 是 | `int`、`float`、`string`、`enum`、`list` |
| `default_val` | string | 是 | 缺失默认值，按 dtype 解析 |
| `role` | string | 否 | `feature`、`label`、`discard`，默认 `feature` |
| `source` | string | 否 | 推理请求分组，常见为 `User`、`Item`、`Context` |
| `data_source` | string | 否 | 引用 `data_sources[].name` |

标签列应使用 `role: label`，不应配置在线 `source`。

## `operators`

`operators` 定义 DAG 节点。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `name` | string | 是 | operator 名，全局唯一 |
| `op_type` | string | 是 | 算子类型 |
| `inputs` | list[string] | 是 | 输入 feature 名 |
| `outputs` | list[string] | 是 | 输出 feature 名 |
| `params` | map | 否 | 算子参数 |
| `embed` | map | 否 | 输出是否进入 embedding |

## `embed`

operator 输出声明 `embed` 后会进入模型 embedding。模型不应重复声明 feature 列表；所有模型从 DAG 的 embeddable features 获取输入规格。

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `vocab_size` | int | 是 | embedding 词表大小 |
| `embed_dim` | int | 是 | embedding 维度 |
| `pooling` | string | 否 | `first`、`flatten`、`mean`、`sum`、`max`，默认 `first` |
| `seq_len` | int | 否 | sequence/flatten pooling 的固定长度 |
| `truncation` | string | 否 | `head` 或 `tail` |

## Operator registry

当前支持 17 个基础算子：

```text
Bucketing, DictMapper, StringParser, JsonExtractList,
ListStringParser, Split, FlatSplit, ExpressionOp, Log1p,
CrossFeature, ListOverlap, SequenceOp, StringConcat,
FeatureHash, PluginOp, ParsedFeatureHash, ConcatHash
```

新增算子需要同时实现 Python 和 Rust，并注册到对应 registry。详细参数见 [Feature operators](../feature_operators.md)。

## Quality and bucket statistics

训练会统计 embedding bucket hit count，并导出 `embedding_bucket_report.yaml`。发布 serving 权重时，零命中 row 会按规则替换为活跃 row 均值，避免输出未训练的随机 embedding。

详细 artifact 说明见 [Artifact Reference](artifacts.md)。
