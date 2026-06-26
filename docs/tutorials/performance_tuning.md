# Performance Tuning

本教程介绍训练读取、特征预处理和 HTTP 推理压测的主要调优入口。

## Goal

知道先看哪些参数、如何验证优化没有破坏一致性。

## Training reader

```bash
--read-chunk-rows 65536
--fast-no-na
--memory-map
```

| 参数 | 说明 |
|---|---|
| `--read-chunk-rows` | pandas chunk 行数 |
| `--fast-no-na` | 关闭 NA 检测 |
| `--memory-map` | 本地未压缩文件启用 memory map |

## Preprocessing

训练侧预处理优化点：

- 减少 list 拷贝。
- 按 feature 列构造 tensor。
- FeatureHash / split/parser 使用有界缓存。
- bucket hit count 用于发布前处理 inactive row。

## Serving benchmark

synthetic smoke 只验证 HTTP 链路；真实性能压测应使用 discover TSV 和 feature config。

```bash
cargo run --bin bench --release -- \
  --url http://127.0.0.1:8080 \
  --model model_gdcn_esmm \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml
```

## Verify

性能优化后至少跑：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/ -q
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## Next

- 操作指南见 [Tune Training Preprocessing](../how_to/tune_training_preprocessing.md)。
- 压测报告见 [HTTP benchmark report](../http_benchmark_report.md)。
