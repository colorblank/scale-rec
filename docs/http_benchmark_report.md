# HTTP 推理压测报告

## 结论

本次 Rust HTTP 推理压测覆盖两个 discover 模型，并对每个模型重复执行 3 轮真实 discover 输入压测：

- `model_gdcn_esmm`
- `model_discover_unimixer`

两者均在 broadcast 模式下通过 300 QPS / 60s 验收。请求形态为 1 个 user/context + 200 个 item candidates，输入来自真实 discover TSV，并由 bench 按 `source: User/Context/Item` 构造 `/predict/broadcast` 请求。

| 模型 | 轮次 | Success | Errors | RPS | 平均 P50 | 平均 P95 | 平均 P99 | 平均 P99.9 | 最差 P99.9 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `model_gdcn_esmm` | 3 | 18000/轮 | 0/轮 | 300 | 8.4 ms | 16.1 ms | 30.6 ms | 68.4 ms | 96.8 ms | 通过 |
| `model_discover_unimixer` | 3 | 18000/轮 | 0/轮 | 300 | 12.2 ms | 20.7 ms | 42.6 ms | 118.7 ms | 236.1 ms | 通过 |

UniMixer 的中位数延迟稳定在 `12.2 ms`，但尾延迟波动高于 GDCN+ESMM。上一轮单次测试出现的 1s 级 P99.9 尖峰未在本次 3 轮复测中复现。

## 测试环境

| 项目 | 值 |
|---|---|
| 测试日期 | 2026-06-01 |
| 平台 | macOS |
| Rust 后端 | Accelerate CPU |
| 服务二进制 | `target/release/server` |
| 压测二进制 | `target/release/bench` |
| Feature config | `examples/feature_config_discover.yaml` |
| 输入数据 | `python/artifacts/demo/discover_train_data.txt` |
| 输入格式 | 38 列 TSV，无 header |
| 请求模式 | broadcast |
| Batch size | 200 candidates/request |
| 目标负载 | 300 QPS / 60s |

## 模型与产物

| 模型 | Model config | 权重 | HTTP model name |
|---|---|---|---|
| GDCN+ESMM | `examples/model_gdcn_esmm.yaml` | `python/artifacts/demo/model_gdcn_esmm.safetensors` | `model_gdcn_esmm` |
| UniMixer | `examples/model_discover_unimixer.yaml` | `python/artifacts/demo/model_discover_unimixer.safetensors` | `model_discover_unimixer` |

## 构建与启动

```bash
cargo build --release --features macos-accelerate --bin server --bin bench
```

```bash
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --feature-config examples/feature_config_discover.yaml \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

服务启动后确认模型已加载：

```bash
curl http://127.0.0.1:8080/health
```

实测返回：

```json
{"models":["model_gdcn_esmm","model_discover_unimixer"],"status":"ok"}
```

## GDCN+ESMM

### 压测命令

```bash
target/release/bench \
  --target http://127.0.0.1:8080 \
  --model model_gdcn_esmm \
  --mode broadcast \
  --concurrency 300 \
  --batch-size 200 \
  --duration-secs 60 \
  --target-qps 300 \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --no-header
```

### 三轮结果

| 轮次 | Scheduled | Success | Errors | RPS | P50 | P95 | P99 | P99.9 | Mean | Min / Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1 | 18000 | 18000 | 0 | 300 | 8.4 ms | 16.1 ms | 28.6 ms | 54.6 ms | 9.7 ms | 6.6 / 86.7 ms |
| Run 2 | 18000 | 18000 | 0 | 300 | 8.3 ms | 16.4 ms | 35.1 ms | 96.8 ms | 9.9 ms | 6.5 / 136.5 ms |
| Run 3 | 18000 | 18000 | 0 | 300 | 8.4 ms | 15.7 ms | 28.0 ms | 53.8 ms | 9.6 ms | 6.6 / 78.6 ms |
| 平均 | 18000 | 18000 | 0 | 300 | 8.4 ms | 16.1 ms | 30.6 ms | 68.4 ms | 9.7 ms | - |

### 原始输出

```text
Run 1
Scheduled:   18000
Success:     18000  Errors: 0  RPS: 300
P50:         8.4 ms
P95:         16.1 ms
P99:         28.6 ms
P99.9:       54.6 ms
Mean:        9.7 ms
Min/Max:     6.6/86.7 ms

Run 2
Scheduled:   18000
Success:     18000  Errors: 0  RPS: 300
P50:         8.3 ms
P95:         16.4 ms
P99:         35.1 ms
P99.9:       96.8 ms
Mean:        9.9 ms
Min/Max:     6.5/136.5 ms

Run 3
Scheduled:   18000
Success:     18000  Errors: 0  RPS: 300
P50:         8.4 ms
P95:         15.7 ms
P99:         28.0 ms
P99.9:       53.8 ms
Mean:        9.6 ms
Min/Max:     6.6/78.6 ms
```

## UniMixer

### 压测命令

```bash
target/release/bench \
  --target http://127.0.0.1:8080 \
  --model model_discover_unimixer \
  --mode broadcast \
  --concurrency 300 \
  --batch-size 200 \
  --duration-secs 60 \
  --target-qps 300 \
  --input-file python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --no-header
```

### 三轮结果

| 轮次 | Scheduled | Success | Errors | RPS | P50 | P95 | P99 | P99.9 | Mean | Min / Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1 | 18000 | 18000 | 0 | 300 | 12.2 ms | 19.5 ms | 30.3 ms | 58.0 ms | 13.3 ms | 10.0 / 83.4 ms |
| Run 2 | 18000 | 18000 | 0 | 300 | 12.2 ms | 21.8 ms | 61.0 ms | 236.1 ms | 14.9 ms | 10.0 / 349.8 ms |
| Run 3 | 18000 | 18000 | 0 | 300 | 12.2 ms | 20.7 ms | 36.4 ms | 62.0 ms | 13.5 ms | 9.9 / 82.7 ms |
| 平均 | 18000 | 18000 | 0 | 300 | 12.2 ms | 20.7 ms | 42.6 ms | 118.7 ms | 13.9 ms | - |

### 原始输出

```text
Run 1
Scheduled:   18000
Success:     18000  Errors: 0  RPS: 300
P50:         12.2 ms
P95:         19.5 ms
P99:         30.3 ms
P99.9:       58.0 ms
Mean:        13.3 ms
Min/Max:     10.0/83.4 ms

Run 2
Scheduled:   18000
Success:     18000  Errors: 0  RPS: 300
P50:         12.2 ms
P95:         21.8 ms
P99:         61.0 ms
P99.9:       236.1 ms
Mean:        14.9 ms
Min/Max:     10.0/349.8 ms

Run 3
Scheduled:   18000
Success:     18000  Errors: 0  RPS: 300
P50:         12.2 ms
P95:         20.7 ms
P99:         36.4 ms
P99.9:       62.0 ms
Mean:        13.5 ms
Min/Max:     9.9/82.7 ms
```

## 验收判断

300 QPS 验收标准：

- `Scheduled=18000`
- `Success=18000`
- `Errors=0`
- `RPS>=295`

两个模型的 3 轮压测均满足全部标准。

## 延迟分析

GDCN+ESMM 三轮表现稳定，平均 P99.9 为 `68.4 ms`，最差 P99.9 为 `96.8 ms`。

UniMixer 的 P50 稳定在 `12.2 ms`，说明常规请求耗时稳定；尾延迟存在更明显波动，Run 2 的 P99.9 达到 `236.1 ms`。该现象更像是 open-loop 压测下的瞬时排队或系统调度抖动，而不是模型整体变慢。

## 注意事项

- HTTP 请求中的 `model` 必须使用 `/health` 返回的模型名，不能使用模型类型名。
- 真实性能结论以带 `--input-file` 和 `--feature-config` 的 discover 输入压测为准。Synthetic smoke 只验证 HTTP 链路。
- bench 的 open-loop 模式按 `--target-qps` 定速发请求，`--concurrency` 当前不限制最大在途请求数；尾延迟包含服务端处理时间和排队时间。
- 不同平台、不同后端、不同构建参数的结果不能直接混用对比。
