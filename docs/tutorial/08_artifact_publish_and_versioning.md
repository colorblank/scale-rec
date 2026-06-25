# 08. 产物发布与版本管理

[目录](README.md) | [上一章](07_evaluation_and_feature_quality.md) | [下一章](09_rust_inference_service.md)

训练真正结束后，系统并不只是留下一份 `safetensors`。

它会把权重、配置、版本元数据和校验信息一起打包，生成一套可以给 Rust 服务直接消费的发布目录。

## 训练产物长什么样

`python/src/train/app/artifacts.py` 负责管理 run 目录和发布目录。典型结构如下：

```text
python/artifacts/demo/
└── model_gdcn_esmm/<run_version>/
    ├── checkpoints/
    ├── serving/
    │   ├── model.safetensors
    │   ├── model.manifest.yaml
    │   └── configs/
    │       ├── feature_config.yaml
    │       └── model_config.yaml
    ├── embedding_bucket_report.yaml
    └── run.manifest.yaml
```

## 两份 manifest 的区别

### run.manifest.yaml

这是训练过程记录：

- 训练到哪一步。
- 哪个 checkpoint 是 best。
- 哪个是 latest。
- 训练过程中保存过哪些 checkpoint。
- 本次 run 的 published 权重路径是什么。
- 完整训练流的 embedding bucket 报告路径是什么。

### model.manifest.yaml

这是给 serving 用的发布契约：

- `model_id`
- `model_version`
- `weights_file`
- `feature_config_file`
- `model_config_file`
- `weights_sha256`
- `feature_config_sha256`
- `model_config_sha256`
- `tasks`
- `label_col_map`
- `metrics`
- `embedding_bucket_report_file`

Rust 服务加载模型时，优先看这份 manifest，而不是裸权重文件。

当前原生 ESMM 的完整 `output_contract` 仍保存在归档的 `model_config_file` 中。serving
manifest 继续记录由训练构建阶段派生出的 `tasks/label_col_map/metrics` 元数据，尚未
单独保存规范化 contract 或 contract 摘要。

## 为什么要算 sha256

`python/src/train/app/manifest.py` 在写 manifest 时会同时写入：

- 权重文件 hash
- feature config hash
- model config hash
- 当前 git commit

这样做的目的不是“好看”，而是防止：

- 权重和配置被拆开复制。
- 线上加载到旧配置。
- 版本回滚时拿错目录。

## best / latest / published 的关系

这三个概念不要混：

- `latest`：最近一次保存。
- `best`：验证指标最优。
- `published`：最终给线上加载的文件。

默认发布通常会选 `best`，但也可以显式发布 `latest` 或某个别名路径。

## 发布权重为什么可能不同于 best checkpoint

checkpoint 用于恢复训练，必须保留训练结束时的原始参数。serving 权重面向未来线上
请求，还需要处理训练期间从未命中的 embedding bucket。

发布流程会对以下算子的零命中 row 使用各自 embedding 表的活跃 row 均值：

- `DictMapper`，包括零命中的 `default_idx`
- `FeatureHash`
- `ParsedFeatureHash`
- `ConcatHash`

这个处理不改变 `vocab_size`、bucket id 或 Python/Rust 哈希协议，只改变最终 serving
safetensors 中的 row 内容。DeepFM 等模型如果为同一特征维护多张 embedding 表，会
分别计算均值。整张表没有任何活跃 bucket 时发布失败。

因此：

- `best.safetensors` / `latest.safetensors` / `checkpoints/*` 是原始训练权重。
- `serving/model.safetensors` 是经过零命中 row 规范化的发布权重。
- `embedding_bucket_report.yaml` 提供该处理的审计依据。

## 发布路径如何决定

`TrainingArtifactManager.from_config()` 会把：

- `artifact_root`
- `model_name`
- `run_version`
- `publish_path`

组合成完整目录。然后 `finalize()` 负责把最终权重复制到发布目录，并写出两个 manifest。

这意味着你可以：

- 只训练不发布。
- 训练后人工挑一个 checkpoint 发布。
- 把发布目录和训练目录分开存。

## 什么时候应该回滚

只要出现以下情况，就应该回滚到旧版本：

- serving manifest 和权重不一致。
- 线上加载后输出明显异常。
- feature config 改了，但下游取数还没跟上。
- 新版本指标退化。

## 发布前检查清单

1. `weights_sha256` 存在且能对上。
2. `feature_config_sha256` 和 `model_config_sha256` 匹配。
3. legacy 模型的 `tasks` 与训练时一致；原生模型的归档 `model_config_file` 包含正确
   的 `output_contract`。
4. `label_col_map` 没写错。
5. `model_version` 和发布目录版本一致。
6. `embedding_bucket_report_file` 存在，且没有整张表零激活。
7. 用 `validate_manifest` 校验 Rust manifest schema 和权重绑定。

下一章讲 Rust 在线推理服务。那部分会说明 registry 怎么加载 manifest、`/predict` 和 `/predict/broadcast` 怎么走，以及版本和别名如何路由。
