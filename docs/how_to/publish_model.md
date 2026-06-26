# Publish a Model

本文档说明训练结束后如何发布推理权重和 manifest。文件格式参考见 [Artifact Reference](../reference/artifacts.md)。

## Default publish path

未显式传入 `--publish-path` 时，训练会发布到当前 run 的 serving 目录：

```text
python/artifacts/demo/<model-name>/<run-version>/serving/
├── model.manifest.yaml
├── model.safetensors
└── configs/
    ├── feature_config.yaml
    └── model_config.yaml
```

## Explicit publish path

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name demo_train \
  --publish-path python/artifacts/demo/model_gdcn_esmm.safetensors \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm \
  --no-header
```

如果发布路径在 run 目录外，manifest 仍会指向当前 run 中归档的 config 副本。跨机器部署时需要同时携带 manifest 中引用的配置文件。

## What gets published

| File | Purpose |
|---|---|
| `model.safetensors` | Rust Candle 加载的权重 |
| `model.manifest.yaml` | Rust serving 推荐加载入口 |
| `configs/feature_config.yaml` | 本次发布绑定的特征配置 |
| `configs/model_config.yaml` | 本次发布绑定的模型配置 |
| `embedding_bucket_report.yaml` | bucket hit count 报告 |

## Inactive embedding rows

发布权重会根据完整训练流的 bucket hit count 处理零命中 row：

- `DictMapper`：零命中 bucket，包括 `default_idx`，替换为活跃 row 均值。
- `FeatureHash` / `ParsedFeatureHash` / `ConcatHash`：零命中 row 替换为活跃 row 均值，但索引空间不变。
- 整个 embedding 表没有活跃 row 时拒绝发布。

这只影响 serving 权重，训练 checkpoint 保持原始权重。

## Verify before serving

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models demo_gdcn_esmm --force-train
```

期望输出：

```text
Overall Consistency Status: PASS
```

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `train.app.main demo` | Training data, feature/model configs, artifact publication and TSV reader flags | [CLI Reference: Train demo](../reference/cli.md#train-demo) |
| `scale_rec_demo.verify_all` | `--models` selects model keys; `--force-train` retrains before comparison | [CLI Reference: Verify all](../reference/cli.md#verify-all) |
