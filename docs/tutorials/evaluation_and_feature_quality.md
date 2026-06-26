# Evaluation and Feature Quality

本教程介绍训练评估、feature quality 和 embedding bucket hit count。

## Goal

理解训练日志中的 loss、metrics、feature quality 和 bucket report。

## Metrics

metrics 来自 model config 的 `output_contract.metrics`：

```yaml
metrics:
  - name: click_auc
    source: click_logit
    label: is_click
    type: auc
```

常见指标：

- `auc`
- `prauc`
- `logloss`
- `mse`
- `mae`

## Feature quality

训练启动会从验证 batch 计算 feature quality，用来观察：

- source 缺失率。
- 默认值命中率。
- sequence 空值率。
- padding rate。
- hash/cache 统计。
- bucket utilization。

## Bucket hit count

训练器会在所有实际反向传播 batch 上累计 embedding bucket 命中次数。结果写入：

```text
embedding_bucket_report.yaml
```

发布 serving 权重时，零命中 embedding row 会被替换为活跃 row 均值，避免线上输出随机 embedding。

## What to watch

| Signal | Meaning |
|---|---|
| `bucket_utilization` 很低 | hash 输入可能为空，或 vocab_size 过大 |
| DictMapper default 命中高 | mapping 覆盖不足或输入格式不一致 |
| padding rate 高 | 序列大量为空或 seq_len 过大 |
| metric 无法计算 | label 为空、正负样本不足或 mask 过滤过多 |

## Next

- 操作指南见 [Inspect Feature Quality](../how_to/inspect_feature_quality.md)。
- artifact 参考见 [Artifact Reference](../reference/artifacts.md)。
