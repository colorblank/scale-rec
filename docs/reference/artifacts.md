# Artifact Reference

本文档描述训练产物、发布产物和 manifest 文件的职责。操作流程见 [产物发布与版本管理](../tutorials/artifact_publishing.md)。

## Run directory

默认训练命令会在 `--artifact-dir/<model-name>/<run-version>/` 下创建 run 目录：

```text
python/artifacts/demo/model_gdcn_esmm/20260526_120000/
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

| 文件 | 说明 |
|---|---|
| `checkpoints/*.safetensors` | 训练 checkpoint 权重 |
| `serving/model.safetensors` | 推理发布权重 |
| `serving/model.manifest.yaml` | Rust serving 推荐加载入口 |
| `serving/configs/feature_config.yaml` | 本次发布归档的特征配置副本 |
| `serving/configs/model_config.yaml` | 本次发布归档的模型配置副本 |
| `embedding_bucket_report.yaml` | 完整训练流 embedding bucket 命中报告 |
| `run.manifest.yaml` | 训练过程记录，不是 serving manifest |

## Serving manifest

`serving/model.manifest.yaml` 是生产推荐加载入口。它绑定权重、特征配置、模型配置和校验信息。

常见字段：

| 字段 | 说明 |
|---|---|
| `schema_version` | manifest schema 版本 |
| `model_id` | 请求中的逻辑模型名 |
| `model_version` | 模型版本；建议使用可排序时间戳 |
| `model_type` | 模型类型，必须和模型配置 YAML 的 `type` 一致 |
| `weights_file` / `weights_sha256` | safetensors 路径和 sha256 |
| `feature_config_file` / `feature_config_sha256` | feature config 路径和 sha256 |
| `model_config_file` / `model_config_sha256` | model config 路径和 sha256 |
| `weight_binding` | Python state_dict key 与 Rust VarBuilder 路径绑定信息 |
| `embedding_bucket_report_file` | bucket 命中报告路径 |

所有相对路径都基于 manifest 所在目录解析。

## Run manifest

`run.manifest.yaml` 记录训练过程和 checkpoint 历史。Rust 服务扫描模型目录时会跳过 `run.manifest.yaml`。

典型用途：

- 找到 latest/best checkpoint。
- 恢复训练前查看 run 元数据。
- 审计训练时使用的配置和评估指标。

## Embedding bucket report

训练器会在所有实际反向传播 batch 上累计 embedding bucket 命中次数。报告按 embedding feature 记录：

| 字段 | 说明 |
|---|---|
| `total_hits` | 总命中次数 |
| `active_buckets` | 命中过的 bucket 数 |
| `inactive_buckets` | 零命中 bucket 数 |
| `bucket_utilization` | `active_buckets / vocab_size` |
| `inactive_bucket_ids` | 零命中 bucket id 列表 |
| `bucket_hits` | 与 bucket id 一一对应的完整命中次数 |

发布权重会根据报告规范化零命中 embedding row：

- `DictMapper`：所有零命中 bucket，包括没有命中的 `default_idx`，替换为该表活跃 row 均值。
- `FeatureHash`、`ParsedFeatureHash`、`ConcatHash`：零命中 bucket 未来仍可能被线上新 key 命中，因此索引空间不变，只替换 row 内容为活跃 row 均值。
- 整张表没有活跃 bucket 时拒绝发布。

训练 checkpoint 保持原始权重不变；只有 serving 发布权重会替换零命中 row。
