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

2026-06-03 重新压测了 UniMixer 的 native CPU 路径，并补充了内部分段 profiler。结论：

- UniMixer 内部单次 100 行推理 `model.total` 约 `8.8 ms`，其中 `pswiglu` 仍是主要耗时段，但 `cached_mixing` 和 `cached_linears` 已被 warmup 移出真实推理路径。
- HTTP 端到端压测在 `300 QPS / 30s / batch-size=200 / concurrency=300` 下得到 `P50 27.5 ms`、`P95 58.8 ms`、`P99 103.3 ms`、`P99.9 156.8 ms`，`Errors=0`。
- 这说明当前优化主要改善了模型内部冷启动和稳定态计算，但端到端 HTTP 延迟仍主要受请求解析、批处理和服务端调度影响；与上一轮 native CPU 的 `P50 27.6 ms` 基本持平。

2026-06-02 补充了不同 CPU 后端的对比压测，并在 Rust 侧优化了 GDCN cross/gate GEMM 与 UniMixer token/block 小矩阵乘路径。结论：

- macOS Accelerate 后端下，UniMixer P50 为 `12.7 ms`，GDCN+ESMM P50 为 `9.3 ms`。
- macOS native CPU 后端下，UniMixer P50 为 `27.6 ms`，比 Accelerate 慢约 `2.2x`；GDCN+ESMM P50 为 `10.4 ms`，与 Accelerate 接近。
- 用户在 Linux native CPU 后端观测到 UniMixer P50 约 `8s`、P99 约 `16s`，而 GDCN+ESMM 仍接近 macOS 结果。该现象与 UniMixer 原先依赖大量 3D batched/small matmul 的后端敏感性一致；当前代码已将相关路径改为更稳定的 2D GEMM loop，但 Linux native 结果仍需在目标机器复测。

## 测试环境

| 项目 | 值 |
|---|---|
| 测试日期 | 2026-06-01 |
| 平台 | macOS |
| Rust 后端 | Accelerate CPU |
| 服务二进制 | `target/release/server` |
| 压测二进制 | `target/release/bench` |
| Feature config | `examples/shared/feature_config_discover.yaml` |
| 输入数据 | `python/artifacts/demo/discover_train_data.txt` |
| 输入格式 | 45 列 TSV，无 header；bench 只读取 feature config 需要的输入列，标签列随文件保留 |
| 请求模式 | broadcast |
| Batch size | 200 candidates/request |
| 目标负载 | 300 QPS / 60s |

## 后端对比

### 构建命令

macOS Accelerate:

```bash
cargo build --release --features macos-accelerate --bin server --bin bench
```

macOS native CPU:

```bash
cargo build --release --bin server --bin bench
```

Linux native CPU:

```bash
cargo build --release --bin server --bin bench
```

Linux MKL CPU:

```bash
cargo build --release --features cpu-mkl --bin server --bin bench
```

### 2026-06-02 单轮对比结果

同参数：`300 QPS / 60s / broadcast / batch-size=200 / concurrency=300`。

| 平台 / 后端 | 模型 | Success | Errors | RPS | P50 | P95 | P99 | P99.9 | Mean | Min / Max | 说明 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| macOS / Accelerate | `model_gdcn_esmm` | 18000 | 0 | 300 | 9.3 ms | 22.3 ms | 41.8 ms | 73.3 ms | 11.5 ms | 6.9 / 114.7 ms | GDCN cross/gate GEMM 融合后 |
| macOS / Accelerate | `model_discover_unimixer` | 18000 | 0 | 300 | 12.7 ms | 27.0 ms | 42.6 ms | 75.8 ms | 14.9 ms | 10.2 / 135.6 ms | UniMixer 小矩阵乘优化后 |
| macOS / native CPU | `model_gdcn_esmm` | 18000 | 0 | 300 | 10.4 ms | 25.6 ms | 43.3 ms | 75.4 ms | 12.8 ms | 7.9 / 121.5 ms | 默认 release，无 Accelerate |
| macOS / native CPU | `model_discover_unimixer` | 18000 | 0 | 300 | 27.6 ms | 65.6 ms | 97.8 ms | 140.6 ms | 33.3 ms | 19.0 / 181.3 ms | 默认 release，无 Accelerate |
| Linux / native CPU | `model_discover_unimixer` | - | - | - | 约 8s | - | 约 16s | - | - | - | 用户环境观测，需按当前优化后代码复测 |

### 后端差异分析

GDCN+ESMM 主要由普通 2D dense GEMM 构成，native CPU 与 Accelerate 的差异较小。UniMixer 对后端更敏感，原因是模型结构包含 token/block 维度上的小矩阵乘：

- `PerTokenSwiGlu`: token-specific `up/gate/down` projection。
- `UniMixingLite`: block-local mixing。
- `UniMixing`: standard block-local mixing。

Rust 实现已做以下优化：

- `PerTokenSwiGlu` 将 `up` 和 `gate` 投影融合成一次投影，再按 hidden 维切分。
- `PerTokenSwiGlu` token projection 从 3D batched matmul 改为 token-wise 2D GEMM loop。
- `UniMixingLite` 和 `UniMixing` 的 local mixing 从 3D batched matmul 改为 block-wise 2D GEMM loop。
- `GatedCrossNetwork` 将 cross/gate 两次 GEMM 融合为一次 GEMM，再按输出维切分。

这些优化不改变 safetensors 权重 key，也不需要重新训练模型。UniMixer Rust/Python 输出一致性已验证，最大 logits 差异为 `4e-08`。

## 模型与产物

| 模型 | Model config | 权重 | HTTP model name |
|---|---|---|---|
| GDCN+ESMM | `examples/models/gdcn_esmm.yaml` | `python/artifacts/demo/model_gdcn_esmm/<run_version>/serving/model.safetensors` | `model_gdcn_esmm` |
| UniMixer | `examples/models/unimixer.yaml` | `python/artifacts/demo/model_discover_unimixer/<run_version>/serving/model.safetensors` | `model_discover_unimixer` |

## 构建与启动

```bash
cargo build --release --features macos-accelerate --bin server --bin bench
```

```bash
RUST_LOG=warn \
target/release/server \
  --model-dir python/artifacts/demo \
  --port 8080 \
  --worker-threads 4 \
  --blocking-threads 64
```

服务按发布 manifest 加载模型、模型配置和特征配置。服务启动后确认模型和版本已加载：

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
  --feature-config examples/shared/feature_config_discover.yaml \
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
  --feature-config examples/shared/feature_config_discover.yaml \
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

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `cargo build --release --features ... --bin ...` | `--release` builds optimized binaries; `--features` selects Candle backend features; `--bin` selects server/bench targets | [Development Reference](../reference/development.md) |
| `target/release/server` | `--model-dir`, `--worker-threads`, `--blocking-threads` configure serving process and runtime | [CLI Reference: Rust server](../reference/cli.md#rust-server) |
| `target/release/bench` | `--target`, `--model`, `--mode`, `--input-file`, `--feature-config`, `--target-qps` and timing flags | [CLI Reference: Rust bench](../reference/cli.md#rust-bench) |
| `curl /health` | HTTP endpoint check | [HTTP API: GET /health](../reference/http_api.md#get-health) |
