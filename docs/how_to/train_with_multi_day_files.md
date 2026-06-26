# Train with Multi-day Files

本文档说明如何用日期范围从 glob 结果中选择多日训练文件。

## Goal

从文件名中解析日期，并按日期升序读取：

```text
data/user_20260325.txt
data/user_20260326.txt
data/user_20260327.txt
```

日期闭区间内缺任意一天会直接报错。

## Command

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 \
  --end-date 20260331 \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 3 \
  --batch-size 1024 \
  --no-header
```

## Validation split

默认行为：

- 前面的日期文件用于训练。
- 最后一个日期文件用于收集验证 batch。
- 训练阶段会跳过最后日期文件中已经作为验证集的 batch。

如果传入 `--eval-data`：

- 所有 `--data-glob` 文件都用于训练。
- 验证集只从 `--eval-data` 读取。

## Incremental fine-tuning

从已有 safetensors 初始化权重：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 \
  --end-date 20260331 \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --init-weights python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  --epochs 3 \
  --batch-size 1024 \
  --no-header
```

`--init-weights` 只加载模型参数，不恢复 optimizer、scheduler、EMA 或 epoch 状态。恢复中断训练应使用 `--resume-from`。

## Next steps

- 独立验证集见 [Train with Independent Eval Data](train_with_independent_eval_data.md)。
- 产物结构见 [Artifact Reference](../reference/artifacts.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `train.app.main discover` | `--data-glob`, `--start-date`, `--end-date` select dated files; `--init-weights` fine-tunes from safetensors | [CLI Reference: Train discover](../reference/cli.md#train-discover) |
