# Inspect Feature Quality

本文档说明如何理解训练侧 feature quality 和 embedding bucket hit count。

## Where reports are written

训练 run 目录会写出：

```text
embedding_bucket_report.yaml
run.manifest.yaml
```

如果使用显式 `--publish-path`，发布权重旁边也会写出同 stem 的 bucket report。

## Bucket report fields

| 字段 | 说明 |
|---|---|
| `total_hits` | 总命中次数 |
| `active_buckets` | 至少命中过一次的 bucket 数 |
| `inactive_buckets` | 零命中 bucket 数 |
| `bucket_utilization` | `active_buckets / vocab_size` |
| `inactive_bucket_ids` | 零命中 bucket id 列表 |
| `bucket_hits` | 每个 bucket 的命中次数 |

## What counts as a hit

统计基于实际进入反向传播的训练 batch。断点续训时，bucket tracker 状态会从 checkpoint 恢复并继续累计。

sequence 特征中的 padding bucket 也会记录命中。当前 embedding pooling 没有 padding mask，因此 padding row 仍可能参与模型输入。

## How inactive rows are handled

发布 serving 权重时：

- `DictMapper` 的零命中 row，包括 `default_idx`，替换为活跃 row 均值。
- `FeatureHash`、`ParsedFeatureHash`、`ConcatHash` 的零命中 row 替换为活跃 row 均值。
- 整张 embedding 表无活跃 row 时拒绝发布。

这样可以避免线上命中新 key 或未激活 key 时输出随机 embedding。

## Common signals

| Signal | Meaning |
|---|---|
| `bucket_utilization` 极低 | hash 输入可能大面积为空，或 vocab_size 过大 |
| `default_idx` 命中很高 | DictMapper mapping 覆盖不足或输入值格式不一致 |
| sequence padding rate 高 | 序列特征大量为空或 max_len 过大 |
| inactive bucket 很多 | 需要关注发布时 row 替换和线上新 key 风险 |

## Related docs

- [Artifact Reference](../reference/artifacts.md)
- [训练评估与特征质量](../tutorials/evaluation_and_feature_quality.md)
