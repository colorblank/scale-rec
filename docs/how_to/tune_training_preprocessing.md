# Tune Training Preprocessing

本文档说明训练侧特征预处理的性能调优入口。完整大文件训练说明见 [性能优化与大文件训练](../tutorials/performance_tuning.md)。

## Reader options

demo 训练使用 pandas chunk 读取 TSV。

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data data/train.tsv \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name demo_train \
  --read-chunk-rows 65536 \
  --fast-no-na \
  --memory-map \
  --no-header
```

| 参数 | 说明 |
|---|---|
| `--read-chunk-rows` | pandas chunk 行数；太小会增加读取开销 |
| `--fast-no-na` | 关闭 pandas NA 检测，适合 NULL 很少的大文件 |
| `--memory-map` | 本地未压缩文件可开启 memory map |
| `--prefetch-batches` | 后台提前预处理 batch，减少训练等待 |

## Python preprocessor optimizations

当前训练侧预处理已包含这些低风险优化：

- DAG batch 执行复用已有 list 输入，避免重复拷贝。
- `DagPreprocessor` 按 embedding feature 列构造 tensor。
- `Bucketing` 使用二分查找。
- `DictMapper` 对字符串输入直接查表。
- `FeatureHash` 使用有界 LRU 缓存 hash 结果。
- `Split`、`ListStringParser`、`FlatSplit`、`ParsedFeatureHash` 使用有界 LRU 缓存解析结果。

这些优化主要降低 Python 层循环、重复字符串解析和重复 hash 计算开销。

## When to consider Rust preprocessing

如果 profile 显示 CPU 时间仍主要消耗在 FeatureHash、DictMapper、Bucketing 或 sequence padding，可以考虑训练侧 Rust preprocessor backend。

启用参数、构建步骤、一致性验证和 benchmark 说明见 [Rust 训练阶段特征预处理](../rust-pretrain-preprocessing.md)。

建议路径：

```text
pandas chunk -> dict[str, list] -> Rust batch preprocessor -> numpy arrays -> torch.from_numpy
```

不要一开始直接让 Rust 依赖 PyTorch Tensor；先返回 numpy 更稳。

## Verification

涉及预处理性能或语义改动时至少执行：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/ -q
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `train.app.main demo` | `--read-chunk-rows`, `--fast-no-na`, `--memory-map` tune TSV reading; data/config/training flags follow demo training | [CLI Reference: Train demo](../reference/cli.md#train-demo) |
| `pytest` | `python/tests/ -q` runs Python tests quietly | [Development Reference](../reference/development.md) |
| `scale_rec_demo.verify_all` | `--models all --force-train` verifies all demo models after retraining | [CLI Reference: Verify all](../reference/cli.md#verify-all) |
