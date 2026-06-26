# Multi-day Training

本教程介绍多日文件训练、独立验证集和增量微调。

## Goal

用日期范围选择训练文件，并控制验证集来源。

## Multi-day command

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 \
  --end-date 20260331 \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 3 --batch-size 1024 --no-header
```

文件名中的第一个 8 位数字作为日期。日期闭区间内缺少任意一天会报错。

## Eval behavior

默认验证集来自最后一天文件：

```text
前 N-1 天 -> train
最后一天前 eval_samples -> eval
最后一天剩余 batch -> train
```

传入 `--eval-data` 后：

```text
所有 data-glob 文件 -> train
eval-data -> eval
```

## Fine-tuning

`--init-weights` 从已有 safetensors 初始化模型参数：

```bash
--init-weights python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors
```

它不恢复 optimizer、scheduler、EMA 或 epoch 状态。

中断恢复使用：

```bash
--resume-from path/to/checkpoint.safetensors
```

## Next

- 操作指南见 [Train with Multi-day Files](../how_to/train_with_multi_day_files.md)。
- 独立验证集见 [Train with Independent Eval Data](../how_to/train_with_independent_eval_data.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `train.app.main demo` | `--data-glob`, `--start-date`, `--end-date`, `--init-weights`, `--resume-from` and common training flags | [CLI Reference: Train demo](../reference/cli.md#train-demo) |
