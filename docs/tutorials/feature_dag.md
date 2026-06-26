# Feature DAG

本教程介绍 Python 训练和 Rust 推理共享的特征 DAG。

## Goal

理解原始字段如何通过 operator DAG 变成模型输入 tensor。

## Feature config

核心文件：

```text
examples/shared/feature_config_discover.yaml
```

顶层结构：

```yaml
version: "1.0.0"
data_sources: []
sources: []
operators: []
```

## Sources

`sources` 定义原始字段、类型、默认值和角色：

```yaml
sources:
  - name: user_id
    source: User
    dtype: int
    default_val: "0"
    role: feature
```

## Operators

`operators` 定义 DAG 节点：

```yaml
operators:
  - name: user_hash
    op_type: FeatureHash
    inputs: [user_id]
    outputs: [user_id_idx]
    params:
      vocab_size: 100000
    embed:
      vocab_size: 100000
      embed_dim: 16
```

## Embeddable features

只有声明了 `embed` 的 operator 输出会进入模型 embedding。模型不重复声明 feature 列表，而是从 DAG 获取 embeddable features。

## Consistency rule

Python 和 Rust 必须：

- 解析同一份 YAML。
- 使用同一套算子语义。
- 对默认值、hash、split、padding、truncation 保持一致。

涉及特征 DAG 的改动必须跑：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## Next

- 算子参考见 [Feature operators](../feature_operators.md)。
- 配置参考见 [Feature Config Reference](../reference/feature_config.md)。
