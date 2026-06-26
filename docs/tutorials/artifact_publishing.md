# Artifact Publishing

本教程介绍训练产物如何发布给 Rust serving。

## Goal

理解 run 产物、serving 产物和 manifest 的关系。

## Run layout

```text
python/artifacts/demo/model_gdcn_esmm/<run-version>/
├── checkpoints/
├── serving/
│   ├── model.manifest.yaml
│   ├── model.safetensors
│   └── configs/
│       ├── feature_config.yaml
│       └── model_config.yaml
├── embedding_bucket_report.yaml
└── run.manifest.yaml
```

## Serving manifest

Rust serving 推荐加载：

```text
serving/model.manifest.yaml
```

它绑定：

- model id / version / type。
- weights path and sha256。
- feature config path and sha256。
- model config path and sha256。
- weight binding。
- embedding bucket report。

## Publishing behavior

训练结束发布时：

1. 选择 best checkpoint 或当前模型参数。
2. 写出 serving safetensors。
3. 根据 bucket report 规范化零命中 embedding row。
4. 复制 feature/model config 到 serving configs。
5. 写出 serving manifest。

## Next

- 操作指南见 [Publish a Model](../how_to/publish_model.md)。
- artifact 参考见 [Artifact Reference](../reference/artifacts.md)。
