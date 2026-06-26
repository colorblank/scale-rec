# Offline Training

本教程介绍 discover TSV 的离线训练流程。

## Goal

跑通一次训练，并理解训练入口如何组合 feature config、model config 和 train config。

## Training command

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm
```

## Config layers

| Config | Purpose |
|---|---|
| `feature_config_discover.yaml` | 原始字段、DAG、embedding、label role |
| `discover_label_policy.yaml` | demo 数据生成规则，不参与训练前向 |
| `train_defaults.yaml` | batch size、optimizer、eval、checkpoint、EMA 等默认值 |
| `examples/models/*.yaml` | 模型结构、output_contract、loss、metrics、outputs |

CLI 参数优先级最高，可以覆盖训练默认值。

## Training loop

训练主流程：

1. pandas chunk 读取 TSV。
2. Python feature DAG 预处理 batch。
3. 构造模型输入 tensor。
4. 模型前向。
5. 按 `output_contract.objectives` 计算 loss。
6. 反向传播并更新参数。
7. 评估 metrics。
8. 导出 checkpoint 和 serving artifact。

## Logs to watch

训练启动时会打印：

- 数据文件数。
- total/train/eval 行数。
- batch 数估计。
- tasks 和 label 映射。
- reader / prefetch / checkpoint 配置。

如果没有 supervised batch，优先检查 label 列和切分摘要。

## Next

- 多日训练见 [Multi-day Training](multi_day_training.md)。
- artifact 见 [Artifact Publishing](artifact_publishing.md)。
