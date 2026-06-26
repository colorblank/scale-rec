# HTTP API

Rust HTTP 服务由 `src/bin/server.rs` 启动，路由定义在 `src/server/routes.rs`。所有接口使用 JSON，请求和响应字段名保持 snake_case。

如果你先看教程，会更容易理解这些接口背后的特征契约和发布约束：

- [09. Rust 在线推理服务](../tutorials/rust_inference_service.md)
- [08. 产物发布与版本管理](../tutorials/artifact_publishing.md)
- [03. 特征工程契约](../tutorials/feature_dag.md)

## 教程对照

API 文档按接口展开，教程按系统链路展开。对照关系如下：

| API 主题 | 对应教程 |
|---|---|
| `/models/{model}/features` | [03. 特征工程契约](../tutorials/feature_dag.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md) |
| `/predict` | [01. 排序系统全链路架构](../tutorials/end_to_end_recommendation.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md) |
| `/predict/broadcast` | [02. 样本表、标签与任务定义](../tutorials/samples_labels_and_tasks.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md) |
| serving manifest / 版本选择 | [08. 产物发布与版本管理](../tutorials/artifact_publishing.md) |
| 错误响应 / 兼容加载 | [08. 产物发布与版本管理](../tutorials/artifact_publishing.md)、[11. Debug 与一致性验证](../tutorials/debugging_consistency.md) |

## 启动服务

对应教程： [08. 产物发布与版本管理](../tutorials/artifact_publishing.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

推荐通过 serving manifest 加载模型：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --port 8080
```

服务默认启用基础过载保护：请求体上限 `8 MiB`、全局 `1000 req/s` 限流、最多 `512` 个并发请求、单请求 `30s` 超时。可通过环境变量或 CLI 调整：

| 配置 | 环境变量 | CLI | 默认值 |
|---|---|---|---:|
| 监听端口 | `SCALE_REC_PORT` | `--port` | `8080` |
| 请求体上限 | `SCALE_REC_MAX_BODY_BYTES` | `--max-body-bytes` | `8388608` |
| 全局限流 | `SCALE_REC_RATE_LIMIT_PER_SECOND` | `--rate-limit-per-second` | `1000` |
| 并发请求上限 | `SCALE_REC_MAX_CONCURRENCY` | `--max-concurrency` | `512` |
| 请求超时 | `SCALE_REC_REQUEST_TIMEOUT_SECS` | `--request-timeout-secs` | `30` |

只加载指定模型或版本：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm.manifest.yaml \
  --port 8080
```

完整加载规则见 [Rust Model Loading](rust_model_loading.md)。

## 通用约定

对应教程： [01. 排序系统全链路架构](../tutorials/end_to_end_recommendation.md)、[08. 产物发布与版本管理](../tutorials/artifact_publishing.md)。

Base URL:

```text
http://127.0.0.1:8080
```

Content type:

```text
application/json
```

版本选择规则：

| 请求字段 | 行为 |
|---|---|
| `model` | 必填，模型逻辑名，对应 serving manifest 中的 `model_id` |
| `version` | 可选，指定 `model_id` 下的具体版本 |
| `fallback_version` | 可选，当 `version` 不存在时尝试回退到该版本 |

未传 `version` 时，服务使用该模型的默认版本。当前默认版本按版本字符串取最大值，因此推荐使用可排序时间戳版本号，例如 `20260526_120000`。

## 接口列表

对应教程： [09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查，返回服务状态和已加载模型 |
| `/metrics` | GET | Prometheus text exposition format 指标 |
| `/models` | GET | 查询全部已加载模型和版本 |
| `/models/{model}` | GET | 查询单个模型的版本信息 |
| `/models/{model}/features` | GET | 查询默认版本的请求特征契约 |
| `/models/{model}/versions/{version}/features` | GET | 查询指定版本的请求特征契约 |
| `/predict` | POST | Pointwise 推理，N 行完整样本得到 N 个预测 |
| `/predict/broadcast` | POST | Broadcast 推理，1 个 user/context 与 N 个 item 组合得到 N 个预测 |

## GET /health

对应教程： [09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

返回服务状态和当前 registry 中的模型信息。

```bash
curl http://127.0.0.1:8080/health
```

## GET /metrics

返回 Prometheus text exposition format，Content-Type 为
`text/plain; version=0.0.4; charset=utf-8`：

```bash
curl http://127.0.0.1:8080/metrics
```

完整指标定义和告警建议见 [Prometheus 指标](prometheus_metrics.md)。

响应：

```json
{
  "status": "ok",
  "models": [
    {
      "name": "model_gdcn_esmm",
      "default_version": "20260526_120000",
      "versions": [
        {
          "version": "20260526_120000",
          "loaded_at": "1780000000",
          "model_type": "gdcn_esmm",
          "manifest_path": "python/artifacts/demo/model_gdcn_esmm.manifest.yaml",
          "is_default": true
        }
      ]
    }
  ]
}
```

## GET /models

对应教程： [09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

查询全部已加载模型。

```bash
curl http://127.0.0.1:8080/models
```

响应：

```json
{
  "models": [
    {
      "name": "model_gdcn_esmm",
      "default_version": "20260526_120000",
      "versions": [
        {
          "version": "20260526_120000",
          "loaded_at": "1780000000",
          "model_type": "gdcn_esmm",
          "manifest_path": "python/artifacts/demo/model_gdcn_esmm.manifest.yaml",
          "is_default": true
        }
      ]
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `name` | 模型逻辑名 |
| `default_version` | 默认版本；模型未加载成功时可能为空 |
| `versions` | 已加载版本列表 |
| `version` | 版本号 |
| `loaded_at` | 加载时间，当前为 Unix timestamp 字符串 |
| `model_type` | 模型类型 |
| `manifest_path` | serving manifest 路径；旧 `.safetensors` 兼容加载时为空 |
| `is_default` | 是否为默认版本 |

## GET /models/{model}

对应教程： [09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

查询指定模型。

```bash
curl http://127.0.0.1:8080/models/model_gdcn_esmm
```

成功响应与 `/models` 中单个模型对象一致：

```json
{
  "name": "model_gdcn_esmm",
  "default_version": "20260526_120000",
  "versions": [
    {
      "version": "20260526_120000",
      "loaded_at": "1780000000",
      "model_type": "gdcn_esmm",
      "manifest_path": "python/artifacts/demo/model_gdcn_esmm.manifest.yaml",
      "is_default": true
    }
  ]
}
```

模型不存在时返回 `404 REGISTRY_ERROR`。

## GET /models/{model}/features

对应教程： [03. 特征工程契约](../tutorials/feature_dag.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

查询模型默认版本的请求特征契约。服务从该版本 serving manifest 指向的 `feature_config_file` 加载契约，因此返回内容与模型权重发布时归档的特征配置一致。

```bash
curl http://127.0.0.1:8080/models/model_gdcn_esmm/features
```

响应：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "data_sources": [
    {
      "name": "user_profile_hbase",
      "kind": "hbase",
      "description": "user profile and behavior features",
      "params": null
    }
  ],
  "required_inputs": [
    {
      "name": "user_id",
      "source": "User",
      "data_source": "user_profile_hbase",
      "dtype": "int",
      "default_val": "0"
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `data_sources` | feature config 顶层声明的数据来源目录，例如画像库、搜索索引、实时统计或向量库 |
| `required_inputs` | 在线推理需要准备的原始 feature source 字段；label/discard 字段不会出现在这里 |
| `source` | 字段业务归属，用于 broadcast 请求拆分 user/context 与 item |
| `data_source` | 字段取数来源名，引用 `data_sources[].name` |
| `dtype` | 字段解析类型，和 feature config 中的 `sources[].dtype` 一致 |
| `default_val` | 字段缺失时 Rust/Python DAG 使用的默认值 |

该接口只暴露契约，不执行外部取数。调用方或请求聚合服务应按 `data_sources` 和 `required_inputs[].data_source` 准备字段，再调用 `/predict` 或 `/predict/broadcast`。

## GET /models/{model}/versions/{version}/features

对应教程： [03. 特征工程契约](../tutorials/feature_dag.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

查询指定版本的请求特征契约。

```bash
curl http://127.0.0.1:8080/models/model_gdcn_esmm/versions/20260526_120000/features
```

响应结构与 `/models/{model}/features` 一致。模型或版本不存在时返回 `404 REGISTRY_ERROR`。

## POST /predict

对应教程： [01. 排序系统全链路架构](../tutorials/end_to_end_recommendation.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

Pointwise 推理。`features` 中每一行都是一个完整样本，服务逐行执行 FeatureDag 后调用模型。

请求：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "fallback_version": "20260526_110000",
  "features": [
    {
      "user_id": 42,
      "item_id": 500
    }
  ]
}
```

只使用默认版本时，可以省略 `version` 和 `fallback_version`：

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'
```

响应：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "predictions": [
    {
      "click": 0.73,
      "cvr": 0.12
    }
  ]
}
```

`version` 是实际使用的版本。请求版本不存在且成功回退时，这里会返回 `fallback_version`。

## POST /predict/broadcast

对应教程： [02. 样本表、标签与任务定义](../tutorials/samples_labels_and_tasks.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

Broadcast 推理。服务会把一个 `user` 特征对象和 `items` 中每个 item 合并为完整样本，输出长度与 `items` 相同。

请求：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "fallback_version": "20260526_110000",
  "user": {
    "user_id": 42,
    "city": "shanghai"
  },
  "items": [
    {
      "item_id": 500
    },
    {
      "item_id": 501
    }
  ]
}
```

curl 示例：

```bash
curl -X POST http://127.0.0.1:8080/predict/broadcast \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","user":{"user_id":42},"items":[{"item_id":500},{"item_id":501}]}'
```

响应：

```json
{
  "model": "model_gdcn_esmm",
  "version": "20260526_120000",
  "predictions": [
    {
      "click": 0.73,
      "cvr": 0.12
    },
    {
      "click": 0.21,
      "cvr": 0.04
    }
  ]
}
```

## 错误响应

对应教程： [08. 产物发布与版本管理](../tutorials/artifact_publishing.md)、[11. Debug 与一致性验证](../tutorials/debugging_consistency.md)。

服务代码中显式返回的业务错误使用 `ApiError` JSON 格式：

```json
{
  "code": "REGISTRY_ERROR",
  "message": "model 'model_gdcn_esmm' version 'missing' not found",
  "request_id": null,
  "model_id": "model_gdcn_esmm",
  "details": {
    "requested_version": "missing",
    "fallback_version": null
  }
}
```

状态码和错误码：

| HTTP 状态码 | `code` | 触发场景 |
|---|---|---|
| 400 | `BAD_REQUEST` | 请求结构或批量输入不合法 |
| 404 | `REGISTRY_ERROR` | 模型或版本不存在 |
| 422 | `FEATURE_ERROR` | 特征预处理失败 |
| 500 | `MODEL_ERROR` | 模型 forward 失败 |
| 500 | `INTERNAL_ERROR` | 推理 worker join 等内部错误 |

Malformed JSON、缺少必填字段等 Axum `Json` extractor 错误由框架默认处理，不会进入 `ApiError` 映射。

版本不存在且指定 fallback 时：

- 如果 fallback 版本存在，服务使用 fallback 并返回成功响应。
- 如果 fallback 版本也不存在，返回 `404 REGISTRY_ERROR`。

## 请求特征格式

对应教程： [03. 特征工程契约](../tutorials/feature_dag.md)、[09. Rust 在线推理服务](../tutorials/rust_inference_service.md)。

特征字段是 JSON object，key 为 feature config 中的 source 名。value 支持数字、字符串、数组等 JSON 类型，具体解析由 `examples/shared/feature_config_discover.yaml` 中的 source dtype 和 operator DAG 决定。

Pointwise 示例：

```json
{
  "user_id": 42,
  "item_id": 500,
  "stay_time": 15.0
}
```

Broadcast 示例中，`user` 通常放 `source: User` 和 `source: Context` 字段，`items` 放 `source: Item` 字段。服务端只负责合并对象；字段角色由 FeatureDag 配置决定。
