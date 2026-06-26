# Debugging and Consistency

本教程介绍 Python/Rust 不一致时的排查顺序。

## Goal

把 mismatch 缩小到数据解析、特征 DAG、权重绑定、模型结构或输出转换中的某一层。

## Start with verify_all

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

## Debug order

1. 确认 feature config 是否相同。
2. 确认 model config 是否相同。
3. 检查 safetensors key 和 shape。
4. 比较单行 DAG 输出。
5. 比较模型 raw output。
6. 确认 binary logit 是否按 serving 规则 sigmoid。

## Weight binding check

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.check_weight_bindings --models all
```

## Reduce scope

先验证最简单模型：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_lr --force-train
```

如果 LR 也失败，优先看特征预处理和权重加载；如果只有复杂模型失败，再看模型层实现。

## Next

- 操作指南见 [Debug Python/Rust Mismatch](../how_to/debug_python_rust_mismatch.md)。
- 特征配置见 [Feature Config Reference](../reference/feature_config.md)。
- 模型配置见 [Model Config Reference](../reference/model_config.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `scale_rec_demo.verify_all` | `--models` scopes verification; `--force-train` regenerates weights | [CLI Reference: Verify all](../reference/cli.md#verify-all) |
| `scale_rec_demo.check_weight_bindings` | `--models all` validates every demo model binding | [CLI Reference: Check weight bindings](../reference/cli.md#check-weight-bindings) |
