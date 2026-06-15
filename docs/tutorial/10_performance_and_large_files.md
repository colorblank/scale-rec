# 10. 性能优化与大文件训练

[目录](README.md) | [上一章](09_rust_inference_service.md) | [下一章](11_debug_and_consistency.md)

当训练数据从 demo 级别变成几十 GB 以后，性能问题通常不在模型，而在数据读取和特征预处理。

这一章只看和大文件相关的几件事：

1. pandas 怎么分块读。
2. 多日文件怎么流式拼接。
3. 什么时候启用 `memory_map` 和 `fast_no_na`。
4. `bench` 压测工具怎么用。

## 训练数据读取是分层的

`python/src/train/app/data.py` 把数据读取拆成两部分：

- `stream_file_batches()`：单文件流式读取。
- `stream_files_batches()`：多文件顺序读取。

这意味着训练时不需要一次把整份数据全读进内存，而是按 batch 递进处理。

## chunk 不是越大越好

`read_chunk_rows` 控制 pandas 每次读多少行。默认策略会根据 `batch_size` 自动推导一个较大的 chunk。

核心原则是：

- chunk 太小，会频繁触发 pandas 开销。
- chunk 太大，会拉高峰值内存。
- 一般保持 chunk 大于等于 batch size，且在本地文件上尽量让读取连续。

如果你明确知道 batch 较大、文件也很规整，可以提高 chunk。

## `fast_no_na` 适合什么场景

`fast_no_na` 会关闭 pandas 的 NA 检测，减少解析开销。

适合以下条件同时成立的情况：

- NULL 很少。
- 默认值处理可以交给 DAG / 预处理层。
- 你更关心吞吐而不是宽松的缺失值识别。

如果数据里本来就有大量空字符串或特殊 NULL 标记，先别急着开。

## `memory_map` 什么时候有用

`memory_map=True` 适合本地未压缩文件。它能减少部分文件读写开销，但并不是所有环境都能明显收益。

经验上：

- 本地单机训练可以尝试。
- 网络盘、压缩文件、远端挂载一般不指望它带来明显提升。

## item 文件索引也是流式的

如果你训练前要构建 item index，`build_item_index()` 会：

- 读取多日 item 文件。
- 用 `item_id` 去重。
- 后读文件覆盖先读文件。
- 只保留 feature-role 的字段。

这样做的目的是让 item 侧特征在跨天时保持“新覆盖旧”的语义。

## 多日训练时的读取顺序

大文件训练时，文件顺序比单文件训练更重要。因为：

- 训练集和验证集按文件切分。
- 后一天会覆盖前一天的 item 信息。
- 日期顺序影响 checkpoint 的可复现性。

所以不要依赖操作系统 glob 的天然顺序，应该通过 `--data-glob`、`--start-date`、`--end-date` 固定住读取顺序。

## `bench` 是干什么的

`src/bin/bench.rs` 是服务压测工具，支持两种模式：

- `pointwise`：直接打 `/predict`。
- `broadcast`：打 `/predict/broadcast`。

它可以：

- 固定并发。
- 固定 duration。
- 设定目标 QPS。
- 从输入文件读真实样本。
- 从 feature config 解析 broadcast 所需的 user / item 结构。

典型用法：

```bash
cargo run --bin bench -- \
  --target http://localhost:8080 \
  --model gdcn_esmm \
  --mode broadcast \
  --concurrency 64 \
  --batch-size 128 \
  --duration-secs 30 \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --no-header
```

## 压测时重点看什么

至少看这几个维度：

- 吞吐：QPS。
- 延迟：p50 / p95 / p99。
- 错误率：是否有 FEATURE_ERROR / MODEL_ERROR。
- 请求大小：batch size 是否和线上一致。
- 模式：pointwise 和 broadcast 的差异。

如果 broadcast 比 pointwise 明显慢，不一定是服务问题，可能是 item 侧特征太重，或者 feature config 里把本来可以共享的计算放到了 item 侧。

## 训练和服务的性能问题不要混着看

训练慢，优先看：

- chunk 读取。
- prefetch。
- feature quality / DAG 复杂度。
- 训练 batch size。

服务慢，优先看：

- broadcast 是否过重。
- manifest 绑定的模型结构是否太大。
- 特征 DAG 是否有可共享的 user-side 子图。

下一章讲 debug 与一致性验证。那部分会把单样本 trace、batch tensor、Python/Rust golden 以及权重 key 排查放在一起。
