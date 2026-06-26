# Train with Independent Eval Data

本文档说明如何使用独立验证文件训练模型。完整参数参考见 [CLI Reference](../reference/cli.md)。

## Goal

使用一份训练文件和一份验证文件：

```text
train.tsv -> 只用于训练
eval.tsv  -> 只用于验证
```

传入 `--eval-data` 后，训练文件不再切分验证样本。

## Requirements

训练文件和验证文件必须满足：

- 文件格式一致。
- 分隔符一致。
- 是否有 header 一致。
- 字段集合一致。
- 字段顺序一致。

无 header 文件会按 `feature_config.sources` 顺序解释列。

## Command

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data data/train.tsv \
  --eval-data data/eval.tsv \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --no-header \
  --epochs 10 \
  --batch-size 1024
```

如果文件带 header，去掉 `--no-header`。

## Behavior

- `--data` 或 `--data-glob` 展开的训练文件全部用于训练。
- `--eval-data` 指定的文件用于验证。
- `--eval-samples` 仍可限制验证样本数量，尤其适合 demo streaming 模式。
- CSV/TSV 与 parquet 不能混用。

## Common failures

| Error | Check |
|---|---|
| column count mismatch | 无 header 文件列数是否一致 |
| columns must exactly match | header 文件列名和顺序是否一致 |
| missing label | label source 是否在 feature config 中声明为 `role: label` |
| no supervised batches | 训练/验证切分、label 列名和 task/output_contract 是否一致 |

## Next steps

- 多日训练见 [Train with Multi-day Files](train_with_multi_day_files.md)。
- 训练参数见 [CLI Reference](../reference/cli.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `train.app.main demo` | `--eval-data` selects an independent validation file; other data/config/training flags follow the demo trainer | [CLI Reference: Train demo](../reference/cli.md#train-demo) |
