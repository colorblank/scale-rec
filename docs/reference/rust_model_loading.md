# Rust Model Loading

Rust 推理服务推荐通过 serving manifest 加载模型。manifest 把权重、特征配置、模型配置和校验信息绑定在一起，是线上发布的权威入口。

## Required files

推荐发布目录：

```text
serving/
├── model.manifest.yaml
├── model.safetensors
└── configs/
    ├── feature_config.yaml
    └── model_config.yaml
```

| 文件 | 是否推荐必备 | 说明 |
|---|---:|---|
| `model.manifest.yaml` | 是 | Rust serving 加载入口 |
| `model.safetensors` | 是 | Candle 加载的模型权重 |
| `configs/feature_config.yaml` | 是 | Rust/Python 共享特征 DAG |
| `configs/model_config.yaml` | 是 | 模型结构和 output_contract |

旧的裸 `.safetensors` 加载方式仍兼容，但需要 `--feature-config` fallback。生产不推荐。

## Load a model directory

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

`--model-dir` 会递归扫描最多 3 层目录，加载 serving manifest，并跳过训练用的 `run.manifest.yaml`。如果扫描到了 serving manifest，服务只按 manifest 加载，不再扫描松散 `.safetensors`。

## Load one manifest

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

`--model-path` 可以重复传入，也可以指向目录。只要传入了 `--model-path`，服务只加载显式路径，不再扫描整个 `--model-dir`。

## Legacy safetensors mode

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  --feature-config examples/shared/feature_config_discover.yaml
```

旧兼容模式下：

- `.safetensors` 文件 stem 作为模型名。
- 版本固定为 `default`。
- 必须提供 `--feature-config`。
- model config 会在权重目录、`--model-dir` 和 feature config 目录中按候选文件名查找。

## Version selection

服务支持同一个 `model_id` 加载多个 `model_version`。请求未指定 `version` 时，服务使用默认版本。默认版本按版本字符串取最大值，因此推荐使用可排序时间戳，例如：

```text
20260526_120000
```

请求可以显式指定版本和 fallback：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "fallback_version": "20260525_120000",
  "features": [{"user_id": 42, "item_id": 500}]
}
```

## Validation at load time

manifest 加载会校验：

- feature config sha256。
- model config sha256。
- safetensors sha256。
- model type 与 model config 的 `type` 一致。
- safetensors key/shape 是否满足模型 `weight_binding`。

如果 key 缺失或 shape 不匹配，服务加载失败。

## Inspect loaded models

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm
curl http://127.0.0.1:8080/models/model_gdcn_esmm/features
```

HTTP 请求响应格式见 [HTTP API](http_api.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `cargo run --bin server` | `--model-dir`, `--model-path`, `--feature-config` and runtime serving flags | [CLI Reference: Rust server](cli.md#rust-server) |
| `curl /health` / `/models` / `/models/{model}` / `/features` | HTTP endpoint paths and response fields | [HTTP API](http_api.md) |
