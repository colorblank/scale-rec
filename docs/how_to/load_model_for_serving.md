# Load a Model for Serving

本文档说明 Rust 推理服务如何加载训练导出的模型。完整参考见 [Rust Model Loading](../reference/rust_model_loading.md)。

## Recommended: load by model directory

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

服务会递归扫描 serving manifest，并跳过训练用的 `run.manifest.yaml`。

## Load one manifest

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

推荐用于精确加载单个模型版本。

## Required files

```text
serving/
├── model.manifest.yaml
├── model.safetensors
└── configs/
    ├── feature_config.yaml
    └── model_config.yaml
```

manifest 中的相对路径基于 manifest 所在目录解析。服务加载时会校验 sha256、model type、weight binding 和 safetensors key/shape。

## Legacy safetensors mode

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  --feature-config examples/shared/feature_config_discover.yaml
```

仅用于兼容旧产物。生产建议使用 manifest。

## Check loaded models

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm
curl http://127.0.0.1:8080/models/model_gdcn_esmm/features
```

## Run prediction

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'
```

完整 HTTP 协议见 [HTTP API](../reference/http_api.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `cargo run --bin server` | `--model-dir`, `--model-path`, `--feature-config`, `--port` and serving limits | [CLI Reference: Rust server](../reference/cli.md#rust-server) |
| `curl /health` / `/models` / `/predict` | HTTP method, endpoint path, JSON body and headers | [HTTP API](../reference/http_api.md) |
