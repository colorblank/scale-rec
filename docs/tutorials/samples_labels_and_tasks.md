# Samples, Labels, and Tasks

本教程说明 discover 样本表如何定义 feature、label、discard，以及模型如何通过 `output_contract` 定义训练目标和指标。

## Goal

理解一行训练样本代表什么，以及 label 如何进入多目标训练。

## Sample row

discover demo 默认输出无 header TSV。列定义来自：

```text
examples/shared/feature_config_discover.yaml
```

`sources` 中的 `role` 决定列用途：

| role | Meaning |
|---|---|
| `feature` | 进入特征 DAG 和模型 |
| `label` | 进入 loss / metrics，不参与在线特征输入 |
| `discard` | 读取后丢弃 |

## Labels

discover 示例包含多列 label，例如：

```text
is_click
is_cvr
is_click_detail
is_click_stock
stay_time_label
```

真实业务接入时，应由业务样本生产链路生成 label；`discover_label_policy.yaml` 只用于 demo 数据生成。

## Tasks with output_contract

当前示例模型统一使用 `output_contract.version: 1`。训练目标写在 model config 的 `objectives` 中：

```yaml
objectives:
  - name: click_loss
    source: click_logit
    label: is_click
    weight: 1.0
    loss:
      type: binary_cross_entropy_with_logits
```

总损失是所有 objective 的加权和：

```text
total_loss = sum(objective_loss * objective.weight)
```

## Metrics

评估指标写在 `metrics` 中：

```yaml
metrics:
  - name: click_auc
    source: click_logit
    label: is_click
    type: auc
```

## Next

- 配置参考见 [Model Config Reference](../reference/model_config.md)。
- 特征列参考见 [Feature Config Reference](../reference/feature_config.md)。
