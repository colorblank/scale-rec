# 04. 离线训练流程

[目录](README.md) | [上一章](03_feature_contract.md) | [下一章](05_multi_day_incremental.md)

这一章把“特征已经对齐之后，训练程序到底怎么跑起来”讲清楚。

核心结论很简单：训练入口不是一个黑盒脚本，而是由 `python/src/train/app/main.py` 组装出来的一条流水线。

```text
FlowConfig / ModelConfig / TrainConfig
  -> FeatureDag / FeatureInfo
  -> Model build
  -> DataFrame / TSV batches
  -> preprocess_batch()
  -> forward -> loss -> backward
  -> evaluator / feature quality
  -> checkpoint / safetensors / manifest
```

## 训练入口有三种模式

`python/src/train/app/main.py` 提供三个子命令：

- `single`：单文件训练，适合 CSV / Parquet。
- `discover`：discover-main-sort 的 TSV 训练入口。
- `all`：同一份数据上批量训练多个模型。

它们共享同一套模型构建、batch 预处理、评估和产物导出逻辑，差异只在于数据来源和模型配置。

最常用的是 `discover`：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header
```

## 配置合并顺序

训练参数不是只看命令行。实际优先级是：

```text
train_defaults.yaml < model YAML < CLI 参数
```

`python/src/train/app/cli.py` 负责把这些值合并成最终的 `TrainConfig`、`ModelConfig` 和 artifact 配置。你在命令行上显式传的值，会覆盖 YAML 默认值。

这意味着：

- 想改训练步数、评估间隔、EMA、early stopping，优先改 `examples/shared/train_defaults.yaml`。
- 想改模型结构和任务定义，改 `examples/models/*.yaml` 中的
  `tasks/task_config` 或 `output_contract`。
- 想临时跑一个实验，直接在 CLI 上覆盖即可。

## batch 是怎么进模型的

训练 batch 的输入不是原始 TSV 直接喂给模型，而是先经过 `TrainingPreprocessor` 包装的 `FeatureDag`：

```text
raw rows
  -> stream_file_batches / stream_files_batches
  -> preprocess_batch()
  -> {feature_name: LongTensor}
  -> model.forward()
```

训练代码里最关键的一行是：

```python
prepared["features"] = preprocessor.preprocess_batch(batch["features"])
```

也就是说，模型只看预处理后的离散索引 tensor，不直接接触原始字符串、JSON 或标签列。

## 一个 epoch 的真实执行顺序

`Trainer.fit()` 的顺序如下：

1. 先用最后一部分数据构建验证集。
2. 再初始化 optimizer、scheduler、EMA。
3. 每个训练 batch 执行前向、loss、反向传播和梯度裁剪。
4. 到达 `eval_interval` 时做一次中间评估。
5. 每个 epoch 结束后做完整评估。
6. 记录 checkpoint，更新 best / latest。
7. 触发 early stopping 则提前结束。
8. 最后导出发布权重和 manifest。

这条链路在 `python/src/train/training/trainer.py` 里是显式代码，不依赖隐式回调。

## 训练时会看哪些指标

`TrainConfig.eval` 决定验证时的指标集合和监控目标。默认会监控 `auc`，但 `monitor_task` 和 `monitor_metric` 都可以改。

训练侧会分别记录三类信息：

- 每个 task 的 loss 和 metrics。
- `feature_quality.*` 的特征质量摘要。
- checkpoint / early stopping / EMA 的状态。

`Evaluator` 会按 `task × metric` 计算结果，`FeatureQualityReport` 会把原始列缺失率、默认值命中率、序列 padding、bucket 利用率等指标整理成数值。

## checkpoint、best、latest 的区别

训练过程中会同时维护三类权重：

- `checkpoints/<version>.safetensors`：每次保存的历史 checkpoint。
- `latest.safetensors`：最近一次保存的别名。
- `best.safetensors`：验证指标最优的别名。

此外还有对应的 `.resume.pt` 状态文件，用于恢复 optimizer、scheduler、EMA、随机数状态和训练进度。

在 `examples/shared/train_defaults.yaml` 里，`keep_checkpoints`、`early_stopping_patience`、`ema_decay` 就是在控制这些行为。

## 一个典型的训练检查点

建议先确认这几个点：

1. `FeatureDag` 能成功构建。
2. `ModelConfig` 能 build 出模型。
3. `tasks` 或 `output_contract.objectives/metrics` 与训练文件里的 label 列一致。
4. `batch_size` 与数据规模匹配。
5. 验证集里确实有监督标签。

如果出现 `No supervised batches were processed`，通常是 label 列名和
`tasks[].label` 或 contract objective label 没对上，或者数据里没有有效标签。

## 推荐理解路径

先把这三个文件串起来看：

- `python/src/train/app/main.py`
- `python/src/train/training/trainer.py`
- `examples/shared/train_defaults.yaml`

再回看 `docs/tutorial/02_samples_labels_tasks.md`，你会更容易理解为什么一个 batch 最终会变成 loss、指标和 checkpoint。

下一章讲多日训练和增量微调，重点是文件顺序、验证集切分以及 `--init-weights` / `--resume-from` 的区别。
