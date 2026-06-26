# Prometheus 指标

Rust 推理服务通过 `GET /metrics` 暴露 Prometheus text exposition format：

```bash
curl http://127.0.0.1:8080/metrics
```

Prometheus scrape 配置示例：

```yaml
scrape_configs:
  - job_name: scale-rec
    metrics_path: /metrics
    static_configs:
      - targets: ["scale-rec:8080"]
```

## 已采集指标

| 指标 | 类型 | Labels | 用途 |
|---|---|---|---|
| `scale_rec_http_requests_total` | Counter | `route,method,status` | HTTP 流量、错误率和限流量 |
| `scale_rec_http_request_duration_seconds` | Histogram | `route,method,status` | HTTP P50/P95/P99 延迟 |
| `scale_rec_inference_requests_total` | Counter | `route,model,status` | 模型请求量及推理错误率 |
| `scale_rec_inference_in_flight` | Gauge | `route,model` | 当前进行中的推理请求 |
| `scale_rec_inference_batch_size` | Histogram | `route,model` | pointwise/broadcast batch 分布 |
| `scale_rec_inference_stage_duration_seconds` | Histogram | `route,model,stage` | `parse/dag/tensor/forward/response` 分阶段耗时 |
| `scale_rec_feature_default_values_total` | Counter | `route,model` | 请求缺字段而使用默认值的次数 |
| `scale_rec_feature_empty_sequences_total` | Counter | `route,model` | 空序列特征次数 |
| `scale_rec_dict_mapper_default_hits_total` | Counter | `route,model,operator` | DictMapper 未命中并回退到 `default_idx` 的元素数 |
| `scale_rec_model_loaded` | Gauge | `model,version,model_type,is_default` | 当前加载的模型版本 |
| `scale_rec_process_start_time_seconds` | Gauge | — | 服务启动时间 |
| `scale_rec_build_info` | Gauge | `version` | 构建版本信息 |

`status` 在 HTTP 指标中是 HTTP 状态码，在推理指标中是 `ok` 或稳定 API 错误码，例如
`BAD_REQUEST`、`REGISTRY_ERROR`、`FEATURE_ERROR`、`MODEL_ERROR`、`INTERNAL_ERROR`。

指标 label 禁止使用 user ID、item ID、请求 ID、原始错误消息、原始特征名等无界值。
请求中的模型名只有在注册表存在该模型时才进入 `model` label；未知模型统一记为
`model="__unknown__"`。

## 通用计算规则

Counter 是进程生命周期内单调递增的累计值，进程重启后归零。使用 `rate()` 或
`increase()` 计算窗口增量，不直接比较两个时刻的原始 Counter。

Gauge 表示采集时刻的状态，可以增加或减少。

Histogram 每次观测一个样本，并同时更新：

- `_bucket{le="X"}`：样本值小于等于 `X` 的累计次数。
- `_bucket{le="+Inf"}`：全部样本次数。
- `_sum`：全部样本值之和。
- `_count`：全部样本次数。

延迟 Histogram 的固定 bucket 单位为秒：

```text
0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25,
0.5, 1, 2.5, 5, 10, 30, +Inf
```

batch size Histogram 的 bucket 为：

```text
1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, +Inf
```

`GET /metrics` 本身不计入普通 HTTP 请求指标，避免 Prometheus scrape 流量污染业务
QPS 和延迟。

## 各指标计算逻辑

### `scale_rec_http_requests_total`

每个完成响应的 HTTP 请求增加 1：

```text
requests_total[route, method, response_status] += 1
```

- `route` 优先使用 Axum 匹配后的路由模板，例如 `/models/{model}`，不会使用包含真实
  模型名的 URL。没有匹配到注册路由时统一记为 `__unmatched__`。
- `method` 为 `GET`、`POST`、`DELETE` 等 HTTP 方法。
- `status` 为最终 HTTP 状态码字符串，例如 `200`、`404`、`422`、`500`。
- 请求体解析失败、路由不存在等由 Axum 生成的响应，只要经过应用路由中间件，也会按
  最终状态码计数。
- 全局限流发生在路由中间件外，单独记录为
  `route="rate_limited",status="429"`。

### `scale_rec_http_request_duration_seconds`

HTTP 请求进入路由指标中间件时开始计时，在获得最终 `Response` 后结束：

```text
duration = response_ready_time - router_middleware_entry_time
```

同一个请求向对应 `route,method,status` Histogram 观测一次。该值包括：

- 模型解析和路由选择。
- `spawn_blocking` 排队及执行时间。
- 特征处理、模型 forward 和响应对象构造。

不包括：

- 客户端下载响应体的网络时间。
- 进入该中间件之前的全局限流等待。
- `/metrics` scrape 请求。

限流请求以 `route="rate_limited"` 记录，当前延迟样本固定为 `0`，只用于统计 429
数量，不应用于业务延迟分析。

### `scale_rec_inference_requests_total`

每个成功解析为 `PredictRequest` 或 `BroadcastRequest` 的推理请求增加 1：

```text
inference_requests_total[route, model, result_status] += 1
```

- `route` 为 `/predict` 或 `/predict/broadcast`。
- `model` 为注册表中的逻辑模型名；不存在的请求模型归并为 `__unknown__`。
- `status="ok"` 表示模型解析、特征处理、forward 和输出转换全部成功。
- 失败状态使用稳定 API 错误码：
  - `REGISTRY_ERROR`：模型、版本、alias 或 fallback 无法解析。
  - `BAD_REQUEST`：特征值无法转换为声明的 dtype。
  - `FEATURE_ERROR`：DAG 执行失败。
  - `MODEL_ERROR`：tensor 构造、模型 forward 或输出转换失败。
  - `INTERNAL_ERROR`：阻塞任务 join 等服务内部失败。

在 JSON 请求体无法反序列化时，处理函数尚未运行，因此只进入 HTTP 指标，不进入该
推理 Counter。

### `scale_rec_inference_in_flight`

推理 handler 接收到合法请求体后立即加 1，handler 以成功或失败返回时通过 RAII guard
减 1：

```text
handler_entry: in_flight[route, model] += 1
handler_exit:  in_flight[route, model] -= 1
```

因此它包含模型解析失败、特征错误和模型错误期间的请求，但不包含 JSON 反序列化失败
或被全局限流直接拒绝的请求。实现会将 gauge 下限限制为 0。

### `scale_rec_inference_batch_size`

每个进入推理 handler 的请求观测一次 batch size，无论最终成功或失败：

```text
/predict:           batch_size = features.len()
/predict/broadcast: batch_size = items.len()
```

空 batch 会观测为 0，并落入第一个 `le="1"` bucket。该指标不区分成功状态；需要分析
成功请求 batch 时，应结合 `scale_rec_inference_requests_total` 判断错误流量占比。

### `scale_rec_inference_stage_duration_seconds`

仅当推理成功并返回完整 `InferenceMetrics` 时记录。引擎内部以微秒计时，导出时除以
`1_000_000` 转换为秒。每个成功请求为五个 stage 各观测一次：

- `stage="parse"`：
  - `/predict`：将每行 JSON 字段转换为类型化 `FeatureValue` 并构造列式输入。
  - `/predict/broadcast`：解析 user 字段、广播 user 值、解析每个 item 字段并构造列式
    batch 的时间总和。
- `stage="dag"`：
  - `/predict`：执行完整预编译特征 DAG。
  - `/predict/broadcast`：user-only 预计算 DAG 与 item batch DAG 两段耗时之和。
- `stage="tensor"`：将 DAG 输出转换为 Candle Tensor，并搬运到当前执行设备。
- `stage="forward"`：调用 Candle 模型 `forward()` 的耗时。
- `stage="response"`：flatten 模型输出、复制为 `Vec<f32>`、执行 logit sigmoid 语义
  转换并构造逐行响应 map 的耗时。

五个 stage 之和通常小于 HTTP 总耗时，因为不包含模型解析、阻塞线程调度、handler
控制逻辑和 Axum 响应封装。

### `scale_rec_feature_default_values_total`

仅在成功推理后，将本次请求的 `default_value_hits` 加到 Counter：

```text
feature_default_values_total[route, model] += default_value_hits
```

pointwise `/predict` 的计算：

```text
default_value_hits =
  Σ 每一行中未提供的 feature source 数量
```

broadcast `/predict/broadcast` 的计算：

```text
default_value_hits =
  Σ 每个 item 中同时未出现在 user 和当前 item 的 source 数量
```

未知字段不属于 DAG source，不会增加 default hit；它也不会覆盖任何 source。该指标的
计数单位是“字段值”，不是“请求数”或“样本数”。例如 100 行请求每行缺少 3 个 source，
本次增加 300。

由于失败请求不会返回完整引擎指标，解析或 DAG 中途失败前产生的 default hit 不计入。

### `scale_rec_feature_empty_sequences_total`

仅在成功推理后，将请求中解析得到的空 `IntList`、`FloatList` 或 `StrList` 数量加到
Counter：

```text
feature_empty_sequences_total[route, model] += empty_sequence_hits
```

- `/predict`：逐行、逐字段检查显式提供的序列值。
- `/predict/broadcast`：user 序列检查一次，每个 item 的序列逐项检查。
- 缺失字段使用默认值时不作为“显式空序列”重复计数；缺失质量由 default 指标表达。
- 非空序列中的 padding `0` 不算空序列。

计数单位是“空序列字段值”。失败请求中已发现的空序列不会导出到 Counter。

### `scale_rec_dict_mapper_default_hits_total`

仅在成功推理后，按配置中的 DictMapper operator name 累加本次 DAG 执行期间的真实映射
未命中次数：

```text
dict_mapper_default_hits_total[route, model, operator] += mapping_miss_count
```

计数发生在 `mapping.get(key)` 查询处，不通过“输出是否等于 `default_idx`”反推。因此：

- 字符串标量 key 不在 mapping 中：增加 1。
- 整数标量先转为十进制字符串再查询；不在 mapping 中增加 1。
- 字符串列表逐元素查询，每个未知元素分别增加 1。
- 不受支持的输入类型直接回退到 `default_idx`：该行增加 1。
- 空列表没有待查询元素，因此增加 0。
- 已知 key 即使配置值恰好等于 `default_idx`，也增加 0。

`operator` label 使用 feature config 中有限集合的 operator `name`，不使用原始 key、
mapping 内容或输入值。

pointwise `/predict` 中，每个样本的 DictMapper 输入都会执行并统计。

broadcast `/predict/broadcast` 中按实际 DAG 执行次数统计：

- user-only DictMapper 在预计算阶段执行一次，因此 user key 未命中增加 1，而不是按 item
  数量放大。
- item/context DictMapper 在候选 batch 阶段逐 item 执行，未知元素按候选数量累加。
- 被 broadcast 预计算跳过的算子不会重复计数。

当多个执行阶段产生同名 operator 统计时，请求内先按 operator name 求和，再累加到
Prometheus Counter。解析、DAG 或模型执行失败的请求不提交本次 operator 统计。

它与 `scale_rec_feature_default_values_total` 的区别：

- `feature_default_values_total`：请求根本没有提供 source 字段。
- `dict_mapper_default_hits_total`：请求提供了值，但值不在 DictMapper mapping 中，或
  输入类型无法映射。

### `scale_rec_model_loaded`

该 Gauge 不保存在指标状态中，而是在每次 `/metrics` scrape 时读取
`ModelRegistry::list_info()` 动态生成：

```text
每个当前已加载的 model/version 输出一条值为 1 的时序
```

Labels：

- `model`：逻辑模型 ID。
- `version`：加载的模型版本。
- `model_type`：`lr/deepfm/mmoe/...` 等结构类型。
- `is_default`：该版本是否为当前默认版本，字符串 `true/false`。

模型卸载或进程重启后，对应时序将从 scrape 结果中消失，而不是输出值 0。

### `scale_rec_process_start_time_seconds`

创建 `PrometheusMetrics` 时读取 Unix epoch 秒：

```text
start_time_seconds = SystemTime::now() - UNIX_EPOCH
```

进程运行期间保持不变。它近似指标子系统启动时间；当前服务启动时立即创建，因此等同于
服务启动时间。

### `scale_rec_build_info`

每次 scrape 固定输出：

```text
scale_rec_build_info{version="<Cargo package version>"} 1
```

`version` 来自编译期 `env!("CARGO_PKG_VERSION")`。该 Gauge 只用于关联部署版本，不表示
运行状态或请求数量。

## 推荐看板

- QPS：`sum(rate(scale_rec_inference_requests_total{status="ok"}[5m])) by (model,route)`
- 错误率：`sum(rate(scale_rec_inference_requests_total{status!="ok"}[5m])) by (model,status)`
  除以对应模型总请求速率。
- P99 总延迟：对 `scale_rec_http_request_duration_seconds_bucket` 使用
  `histogram_quantile(0.99, ...)`。
- P99 forward 延迟：对
  `scale_rec_inference_stage_duration_seconds_bucket{stage="forward"}` 使用
  `histogram_quantile(0.99, ...)`。
- 默认值比例：默认值增量除以 batch 样本量；持续升高通常表示上游字段缺失。
- in-flight：结合 `max_concurrency` 判断是否接近服务并发上限。

## 建议告警

生产阈值应按模型基线校准，建议至少配置：

- 5 分钟推理错误率超过 1%。
- P99 HTTP 或 forward 延迟连续 10 分钟超过模型 SLO。
- `scale_rec_inference_in_flight` 持续接近 `max_concurrency`。
- `FEATURE_ERROR`、`MODEL_ERROR` 或 `INTERNAL_ERROR` 在 5 分钟内出现。
- 默认值或空序列速率相对过去一小时基线上升 3 倍。
- DictMapper default hit 比例相对历史基线上升，或低基数枚举出现持续未知 key。
- 预期模型版本的 `scale_rec_model_loaded` 缺失，或默认版本发生非计划变化。
- `rate_limited` 路由的 HTTP 429 持续出现。

## 部署层应补充的指标

以下指标不由应用重复实现，应由 Kubernetes、node-exporter 或容器运行时采集：

- CPU 使用率、RSS、OOM、线程数、文件描述符。
- Pod 重启、ready 状态、网络吞吐和连接数。
- Candle 后端设备利用率及显存；GPU/Metal/CUDA 使用对应平台 exporter。
- 模型文件加载耗时和热更新失败次数可在后续热更新控制面接入时补充。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `curl /metrics` | GET `/metrics` returns Prometheus text exposition | [HTTP API: GET /metrics](http_api.md#get-metrics) |
