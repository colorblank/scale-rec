# 09. Rust 在线推理服务

[目录](README.md) | [上一章](08_artifact_publish_and_versioning.md) | [下一章](10_performance_and_large_files.md)

训练侧完成之后，Rust 侧负责把同一份特征 DAG 和同一组权重变成低延迟 HTTP 推理服务。

这章的目标是理解服务端的三层结构：

```text
manifest / registry
  -> FeatureDag / DagExecutor
  -> InferenceEngine
  -> Axum routes
```

## 服务从哪里加载模型

`src/server/registry.rs` 会从模型目录或 manifest 目录加载：

- feature config
- model config
- safetensors 权重
- 权重绑定规则
- 版本、别名和路由策略

如果加载的是 manifest，服务会先校验：

- feature config 的 sha256
- model config 的 sha256
- weights 的 sha256
- manifest schema 版本

这一步的意义是把“能不能加载”前置到启动阶段，而不是等到第一条请求才发现配置错了。

## HTTP 路由有哪些

`src/server/routes.rs` 暴露的核心接口是：

- `GET /health`
- `GET /models`
- `GET /models/{model}`
- `GET /models/{model}/features`
- `GET /models/{model}/aliases`
- `POST /models/{model}/aliases/{alias}`
- `DELETE /models/{model}/aliases/{alias}`
- `GET /models/{model}/routing`
- `POST /models/{model}/routing`
- `GET /models/{model}/versions/{version}/features`
- `POST /predict`
- `POST /predict/broadcast`

其中 `features` 接口很重要，它告诉调用方当前模型需要哪些字段，字段从哪里取，线上取数契约怎么拼。

## /predict 和 /predict/broadcast 的区别

### /predict

pointwise 预测。请求里传一批完整样本，每个样本都是一个独立 user-item-context 组合。

### /predict/broadcast

broadcast 预测。请求里传一个 `user`，再传多个 `items`，服务会先把 user 侧特征预计算一次，再对每个 item 逐条打分。

对于召回后排序，这个接口通常更省时。

## 请求是怎么走到模型的

`InferenceEngine` 的执行顺序大致是：

1. 把 JSON row 转成列式数据。
2. 执行 DAG，得到中间特征。
3. 把 embeddable feature 转成 tensor。
4. 调 Candle 模型 forward。
5. 把输出整理成响应。

在这个过程中，`parse_us`、`dag_us`、`tensor_us`、`forward_us`、`response_us` 都会记录下来，方便你定位慢点在什么阶段。

## 版本、别名和回退

`ModelRegistry` 支持三种选版本方式：

- 显式 `version`
- `alias`
- `fallback_version`

如果用户请求的版本或别名不存在，可以用 fallback 版本兜底。

此外还支持路由策略：

- 固定版本
- 按权重分流

这让灰度发布和回滚都可以在服务层完成。

## 为什么线上离线要用同一份 feature config

Rust 服务不会自己猜特征应该怎么处理。它只执行配置里写明的 DAG。

所以这类问题必须从一开始就避免：

- 训练用了一个 hash 配置，服务用了另一个。
- 训练里某个 feature 走了 flatten，服务里却还是 first。
- 训练时有默认值，服务取数时却少了字段。

最稳妥的做法，是把发布目录里的 `feature_config.yaml` 当成线上唯一真相。

## 服务启动前后的检查

启动前：

1. manifest 能被解析。
2. 权重文件存在。
3. feature config 和 model config 的 hash 匹配。

启动后：

1. `/health` 返回 ok。
2. `/models` 能列出模型。
3. `/models/{model}/features` 和训练侧理解一致。
4. `POST /predict` 的输出和 Python 验证一致。

下一章会讲性能优化和大文件训练。那部分主要关心 pandas 分块、memory map、prefetch 和 bench。
