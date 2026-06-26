# Rust Inference Service

本教程介绍 Rust HTTP 推理服务的模型加载和请求形态。

## Goal

启动服务，查看模型，执行 pointwise 和 broadcast 推理。

## Start service

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

或者加载单个 manifest：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

## Inspect models

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm
curl http://127.0.0.1:8080/models/model_gdcn_esmm/features
```

## Predict

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'
```

## Broadcast predict

```bash
curl -X POST http://127.0.0.1:8080/predict/broadcast \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","user":{"user_id":42},"items":[{"item_id":500},{"item_id":501}]}'
```

## Observability

Prometheus metrics:

```bash
curl http://127.0.0.1:8080/metrics
```

## Next

- 加载规则见 [Rust Model Loading](../reference/rust_model_loading.md)。
- HTTP 协议见 [HTTP API](../reference/http_api.md)。
- 指标见 [Prometheus Metrics](../reference/prometheus_metrics.md)。
