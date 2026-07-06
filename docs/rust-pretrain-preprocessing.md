# Rust 训练阶段特征预处理

本文说明如何在 Python 训练阶段启用 Rust 实现的特征预处理。该路径通过 PyO3 扩展 `feat_engine` 调用 `scale-rec-core` 中的 DAG 构建、算子执行和 pooling/padding 逻辑，使训练 batch 预处理与 Rust 推理侧尽量复用同一套实现。

适用场景：

- 训练 profile 显示 CPU 时间主要消耗在特征 DAG、hash、字符串解析、序列 padding。
- 希望训练预处理语义与 Rust serving 更接近，减少 Python/Rust 算子实现漂移。
- 大 batch 或多日流式训练中，Python 预处理成为 GPU 前置瓶颈。

不适用场景：

- 当前瓶颈在数据读取、模型 forward/backward、评估指标或 IO。
- 需要 Python debug tracer 的逐算子可视化输出；Rust 路径只替换 `preprocess_batch()` 热路径。
- 尚未构建本机 `feat_engine` 扩展，且训练任务要求不可回退。

## 架构

```
                          workspace root (Cargo.toml)
                          ├── scale-rec (根 crate: 推理服务器)
                          │   ├── src/feats/mod.rs → re-export from core
                          │   ├── src/feats/dag.rs, metrics.rs, debug/
                          │   ├── src/layers/, src/models/, src/server/
                          │   └── src/bin/
                          │
                          ├── crates/core/ (scale-rec-core)
                          │   ├── feats/config.rs, builder.rs, executor.rs
                          │   ├── feats/defaults.rs, schema.rs, feature_info.rs
                          │   ├── feats/ops/ (17 算子 + registry)
                          │   └── feats/tensor_utils.rs (pooling 共享)
                          │
                          └── python/rust_feat_bridge/ (excluded from workspace)
                              ├── Cargo.toml          # cdylib, 依赖 scale-rec-core + pyo3
                              ├── pyproject.toml       # maturin 构建, Python 包名 feat_engine
                              └── src/lib.rs           # FeatSession PyO3 class
```

## 训练数据流

```
训练:
  TSV / 多日文件
    → Python reader 切 batch
    → {source_col: [raw_value, ...]}
    → FeatureDag.preprocess_batch(rows_or_columns)
    → Python 将 source 值转成 str | None
    → FeatSession.preprocess_batch(columns)
      ├── parse_string_to_fv()     按 SourceDef.dtype 解析
      ├── DagExecutor.execute_plan Rust DAG 执行
      └── feature_column_to_vec()  Rust pooling/padding
    → {feature_name: list[int] | list[list[int]]}
    → torch.tensor(dtype=torch.long)
    → model.forward()

推理 (不变):
  JSON → rows_to_columns()
    → executor.execute_plan()         ← 同一份 plan 代码
    → feature_column_to_tensor()       ← 同一份 pooling 逻辑
    → Candle Tensor → model.forward()
```

Rust 训练预处理只接管特征 batch 预处理，不改变以下部分：

- 训练文件读取、shuffle、batch 切分、label 解析仍在 Python 侧。
- 模型训练、loss、optimizer、评估和 artifact 发布仍在 Python 侧。
- `FeatureDag` 的 Python metadata 仍会构建，供模型构建、特征维度推导、bucket tracker、导出逻辑使用。

## FeatSession

`FeatSession` 是 PyO3 暴露给 Python 的 Rust 会话对象。它在初始化时读取 feature YAML、构建 DAG，并缓存 embeddable features 的输出列位置；训练时对每个 batch 调用 `preprocess_batch()`。

```python
from feat_engine import FeatSession

# 初始化（仅一次，构建 DAG + 推导 FeatureSpec）
session = FeatSession("feature_config.yaml")

# 批量预处理
result = session.preprocess_batch({
    "user_id": ["123", "456"],
    "item_id": ["789", "101112"],
    "scene":   ["1",   "2"],
})
# → {"user_id_idx_a": [1968, 3015], "item_id_idx": [2452, 4820], ...}
```

输入列值必须是 `str | None`，Rust 侧按 `SourceDef.dtype` 严格解析：

- `dtype: int` → `s.parse::<i32>()`，失败用 `default_val`
- `dtype: float` → `s.parse::<f32>()`，失败用 `default_val`
- `dtype: string` → 保留为字符串

返回值只包含 feature config 中可 embedding 的输出特征：

- 标量特征返回 `list[int]`，Python 侧转为 shape `[batch]` 的 `torch.long`。
- 序列特征返回 `list[list[int]]`，Rust 侧已经按 `pooling` / `seq_len` / `truncation` 完成裁剪和 padding，Python 侧转为 shape `[batch, seq_len]` 的 `torch.long`。
- 非 embedding 的中间节点不会返回给模型。

## 构建 feat_engine

从仓库根目录执行：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python maturin develop \
  --manifest-path python/rust_feat_bridge/Cargo.toml --uv
```

如果 CI 或内网环境禁止访问 crates.io，但依赖已经缓存且 `python/rust_feat_bridge/Cargo.lock` 是最新的，可以使用 Cargo 离线模式：

```bash
CARGO_NET_OFFLINE=true PYTHONPATH=python/src:$PYTHONPATH \
  uv run --project python maturin develop \
  --manifest-path python/rust_feat_bridge/Cargo.toml --uv
```

构建完成后，当前 uv 环境中应能导入 `feat_engine`：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python python - <<'PY'
import feat_engine
print(feat_engine.FeatSession)
PY
```

如果构建失败，先确认：

- Rust toolchain 可用，`cargo check -p scale-rec-core` 可以通过。
- 当前 Python 环境由 `uv` 管理，命令在仓库根目录执行。
- `maturin` 来自 `python/pyproject.toml` 的 dev 依赖，因此构建命令需要带 `--project python`。
- `python/rust_feat_bridge/Cargo.toml` 依赖的 `scale-rec-core` 路径仍指向本仓库 `crates/core`。

## 训练入口启用

训练 CLI 通过两个参数控制 Rust 预处理：

| 参数 | 行为 |
|---|---|
| `--use-rust-preprocess` | 尝试启用 Rust `feat_engine`。如果 import 或初始化失败，默认回退到 Python DAG，并写 warning log。 |
| `--require-rust-preprocess` | 强制启用 Rust 预处理。该参数隐含 `--use-rust-preprocess`；如果 `feat_engine` 不可用或初始化失败，训练直接失败。 |

推荐在本地验证、生产训练和性能对比时使用 `--require-rust-preprocess`，避免因为扩展缺失而悄悄回到 Python 路径。

单文件 demo 训练示例：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name demo_train_rust_preprocess \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 \
  --batch-size 1024 \
  --no-header \
  --use-rust-preprocess \
  --require-rust-preprocess
```

多日流式训练示例：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 \
  --end-date 20260331 \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name multi_day_rust_preprocess \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 3 \
  --batch-size 2048 \
  --read-chunk-rows 65536 \
  --prefetch-batches 2 \
  --fast-no-na \
  --no-header \
  --require-rust-preprocess
```

`python/src/train/app/main.py` 会根据上述参数创建 `FeatureDag`：

```python
FeatureDag(
    flow_config,
    use_rust=args.use_rust_preprocess or args.require_rust_preprocess,
    config_path=args.feature_config,
    require_rust=args.require_rust_preprocess,
)
```

## 代码中直接使用

训练主入口之外，也可以在测试、benchmark 或自定义训练脚本中直接构建 `FeatureDag`：

```python
from train.core.config import FlowConfig
from train.core.dag import FeatureDag

flow_config = FlowConfig.from_yaml("examples/shared/feature_config_demo.yaml")
dag = FeatureDag(
    flow_config,
    use_rust=True,
    config_path="examples/shared/feature_config_demo.yaml",
    require_rust=True,
)

tensors = dag.preprocess_batch(
    {
        "user_id": ["1001", "1002"],
        "item_id": ["2001", "2002"],
        "scene": ["1", "2"],
    }
)
```

`preprocess_batch()` 接受两种输入：

| 输入形式 | 说明 |
|---|---|
| `list[dict]` | 行式样本。Python 会先转成 DataFrame/列式数据，再交给预处理路径。 |
| `dict[str, list]` | 列式 batch。训练热路径和 benchmark 推荐使用这种形式，减少行列转换成本。 |

## 数据契约和边界行为

Rust 路径以 feature YAML 的 `sources` 为唯一输入 schema：

- 只会读取 `sources` 中声明的列；额外列会在 Python `FeatureDag` 包装层过滤掉。
- 缺失列由 Rust `DagExecutor` 按 source `default_val` 补齐。
- 列内 `None` 会按 source 默认值处理。
- Python 侧会在调用 Rust 前执行 `str(v) if v is not None else None`，因此输入中的 int/float 会以字符串形式进入 Rust parser。
- 解析失败使用 source 默认值，不中断训练；如果需要发现脏数据，应结合 feature quality、样本检查或 debug 工具。
- 所有输入列长度必须一致；训练 reader 和 batch 构造逻辑应保证这一点。

与 Python 路径相比，语义上需要重点关注：

| 项目 | Rust 训练预处理行为 |
|---|---|
| dtype 解析 | 使用 `scale_rec_core::feats::defaults::parse_string_to_fv()`，与推理侧 typed parsing 对齐 |
| DAG 构建 | 使用 `scale_rec_core::feats::builder::DagBuilder` |
| DAG 执行 | 使用 `scale_rec_core::feats::executor::DagExecutor` |
| pooling/padding | 使用 `scale_rec_core::feats::tensor_utils::feature_column_to_vec()` |
| 输出 dtype | Python 包装层统一转成 `torch.long` |
| debug tracer | Rust 路径不输出 Python tracer 的逐样本 trace |

## 与 prefetch 配合

`--prefetch-batches` 会在线程池中提前调用 `preprocessor.preprocess_batch()`。启用 Rust 路径后，`FeatSession` 在 Rust 执行阶段会释放 GIL，因此 prefetch 能减少训练 loop 等待预处理的时间。

建议：

- CPU 核数充足且模型训练等待数据时，设置 `--prefetch-batches 1` 或 `2`。
- 如果数据读取或模型训练已经占满 CPU，prefetch 过高可能增加上下文切换和内存占用。
- 大 batch 下优先调大 `--read-chunk-rows`，避免 reader 切片过碎。
- 对本地未压缩大文件可同时使用 `--memory-map`；NULL 很少时可使用 `--fast-no-na`。

## 验证一致性

首次启用或修改算子后，至少执行一致性测试：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  pytest python/tests/test_rust_pretrain_consistency.py -v
```

该测试使用相同 feature config 和输入 batch，比较 Rust `FeatSession` 与 Python `FeatureDag` 的 embeddable feature 输出。

涉及算子、pooling、padding 或 feature config schema 的改动时，建议同时执行：

```bash
cargo test -p scale-rec-core
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/test_dag.py -q
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/test_rust_pretrain_consistency.py -q
```

如果改动可能影响 serving 语义，再执行端到端一致性：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## 性能 benchmark

预处理吞吐基准脚本模拟训练热路径：batch 切片、列转 `list`、`FeatureDag.preprocess_batch()`，以及 Python 侧 `torch.tensor()` 构造。脚本默认使用训练入口相同的 null markers：`NULL`、`\N`、`null`、`None`、空字符串，并在进入 Python/Rust 预处理前转成 `None`。

对比 Python 与 Rust：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.benchmark_preprocess \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --mode both \
  --batch-sizes 128,512,1024 \
  --repeat 3 \
  --warmup-batches 2 \
  --no-header \
  --require-rust \
  --profile
```

只跑 Rust：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.benchmark_preprocess \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --mode rust \
  --batch-sizes 1024,2048 \
  --repeat 5 \
  --warmup-batches 2 \
  --no-header \
  --require-rust \
  --profile
```

关键指标：

| 指标 | 含义 |
|---|---|
| `prepare_s` | Python reader/batch 切片后构造 `dict[str, list]` 的耗时 |
| `preprocess_s` | `FeatureDag.preprocess_batch()` 总耗时，包含 Rust/Python DAG、PyO3 边界和 tensor 构造 |
| `rust_parse_s` | Rust 将 `str | None` 按 source dtype 解析为 `Fv` 的耗时 |
| `rust_execute_s` | Rust DAG 执行耗时 |
| `rust_extract_s` | Rust 提取 embeddable features 并执行 pooling/padding 的耗时 |
| `rows/s` | 端到端预处理吞吐 |
| `op:*` / `op_type:*` | `--profile` 下每个 operator 或 operator type 的耗时 |

解读建议：

- `prepare_s` 高：优先调 reader、`--read-chunk-rows`、输入格式和列式 batch 构造。
- `rust_parse_s` 高：检查原始列是否有大量复杂字符串转换，或是否能减少 source 数量。
- `rust_execute_s` 高：看 `op_type:*` 热点，优先优化高频 hash、JSON、split、sequence 算子。
- `rust_extract_s` 高：检查序列特征 `seq_len`、padding 数量和 embedding feature 数量。
- Rust 和 Python 差距小：瓶颈可能不在 DAG，继续看训练 loop、模型或 IO。

## 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: feat_engine` | 先执行 `uv run --project python maturin develop --manifest-path python/rust_feat_bridge/Cargo.toml --uv`。生产训练使用 `--require-rust-preprocess` 让问题显式失败。 |
| `config_path is required when use_rust=True` | 直接构造 `FeatureDag` 时必须传 feature YAML 路径；训练 CLI 会自动传 `args.feature_config`。 |
| 训练日志显示 fallback 到 Python DAG | 检查 `feat_engine` 是否构建在当前 uv 环境，以及初始化时 feature YAML 是否能被 Rust parser 解析。 |
| Rust/Python 输出不一致 | 先跑 `python/tests/test_rust_pretrain_consistency.py`，定位到具体 feature 后检查对应 Rust/Python operator 实现、默认值、padding/truncation。 |
| benchmark 中小 batch Rust 不明显更快 | PyO3 边界和 `torch.tensor()` 构造有固定成本；增大 batch size 或启用 prefetch 后再比较端到端训练吞吐。 |
| debug tracer 没有 Rust operator 细节 | Rust 路径不走 Python tracer；使用 benchmark `--profile` 查看 Rust operator timings，或临时关闭 Rust 路径用 Python debug。 |

## 实现约束

- `python/rust_feat_bridge` 不加入根 workspace，避免常规 Rust 构建默认编译 Python 扩展。
- `feat_engine` 只依赖 `scale-rec-core` 的特征配置、DAG、算子和 tensor 工具，不依赖 PyTorch 或 Candle。
- 返回 Python list 后再由 Python 包装层创建 `torch.Tensor`，避免 bridge 引入 PyTorch C++/Rust 绑定。
- `FeatureDag` 中 Python DAG 始终构建；Rust 只替换训练热路径 `preprocess_batch()`。
- 新增 operator 时必须同时保证 Rust registry、Python registry 和一致性测试覆盖。

### 预处理优化记录

固定 benchmark 参数：demo 数据 2000 行，`--mode rust --batch-sizes 128,512,1024 --repeat 5 --warmup-batches 2 --no-header --require-rust --profile`，Windows dev build。

| 轮次 | 修改 | batch=128 rows/s | batch=512 rows/s | batch=1024 rows/s | 结论 |
|------|------|------------------|------------------|-------------------|------|
| baseline | profile-only，无算子优化 | 3247.4 | 3360.0 | 3466.6 | `ListOverlap`/`StringParser`/`FeatureHash` 为主要热点 |
| 1 | `ListOverlap` 去掉每行两个 `HashSet` 分配，改为小列表无分配扫描 | 3692.6 (+13.7%) | 3992.2 (+18.8%) | 3945.7 (+13.8%) | 保留 |
| 2 | `StringParser` 去掉 `collect::<Vec<&str>>()`，用 `nth(key_index)`，达到 `pad_len` 后提前停止 | 3977.3 (+7.7%) | 4179.6 (+4.7%) | 4078.5 (+3.4%) | 保留 |
| 3 | `FeatureHash` 对 namespace/salt/version 前缀直接按 bytes 参与 DJB2，避免每次 `format!` scoped key | 4022.8 (+1.1%) | 4285.8 (+2.5%) | 4236.6 (+3.9%) | 保留 |
| 4 | `ExecutionPlan` 单输出列直接 move 结果，避免对每个输出列都 clone；默认 `CustomOp::process_batch()` 复用行缓冲，避免 fallback 算子每行分配 `Vec<Fv>` | 未同机复测 | 未同机复测 | 未同机复测 | 保留，属于低风险内存分配优化 |

相对 baseline，当前保留优化后的总提升：batch 128 `+23.9%`，batch 512 `+27.6%`，batch 1024 `+22.2%`。

当前本机校验（Darwin arm64 dev build，demo 数据 2000 行，`--mode rust --batch-sizes 128,512,1024 --repeat 5 --warmup-batches 2 --no-header --require-rust --profile`）：

| case | rows/s | total_s | preprocess_s | 主要热点 |
|-------|--------|---------|--------------|----------|
| baseline batch=128 | 8996.1 | 0.2223 | 0.2132 | `FeatureHash` 0.0675s、`JsonExtractList` 0.0490s、`StringParser` 0.0240s |
| baseline batch=512 | 9270.4 | 0.2157 | 0.2116 | `FeatureHash` 0.0662s、`JsonExtractList` 0.0486s、`StringParser` 0.0238s |
| baseline batch=1024 | 9562.0 | 0.2092 | 0.2061 | `FeatureHash` 0.0660s、`JsonExtractList` 0.0485s、`StringParser` 0.0238s |
| final batch=128 | 9629.9 | 0.2077 | 0.1977 | `FeatureHash` 0.0498s、`JsonExtractList` 0.0494s、`StringParser` 0.0241s |
| final batch=512 | 10302.5 | 0.1941 | 0.1899 | `JsonExtractList` 0.0491s、`FeatureHash` 0.0485s、`StringParser` 0.0239s |
| final batch=1024 | 10449.4 | 0.1914 | 0.1882 | `JsonExtractList` 0.0488s、`FeatureHash` 0.0483s、`StringParser` 0.0237s |

这个结果只用于确认当前热区，不与 Windows dev build 基线直接计算百分比。

本轮 Darwin arm64 连续优化记录：

| 轮次 | 修改 | before rows/s (128/512/1024) | after rows/s (128/512/1024) | 结论 |
|------|------|-------------------------------|------------------------------|------|
| A | `ExpressionOp` 实现专用 `process_batch()`，复用 Rhai `Scope` 和变量名，避免默认 batch fallback 每行重建输入向量和变量名 | 8996.1 / 9270.4 / 9562.0 | 8995.4 / 9450.0 / 9573.6 | 保留；总吞吐基本持平到小幅提升，`ExpressionOp` 从约 0.0097/0.0094/0.0093s 降到 0.0087/0.0084/0.0082s |
| B | `FeatureHash` 标量单输入 cache hit 避免构造 key；单输入 list 直接逐元素 hash，避免每行中间 `Vec<String>` | 8995.4 / 9450.0 / 9573.6 | 9807.5 / 10367.4 / 10517.9 | 保留；`FeatureHash` 从约 0.066s 降到 0.048s |
| C | `FeatSession` 初始化时缓存 source default，`parse_columns()` 复用缓存 | 9807.5 / 10367.4 / 10517.9 | 9764.6 / 10265.1 / 10428.3 | 回滚；三档 batch 均无提升 |
| D | `ListStringParser` 去掉 `collect::<Vec<&str>>()`，用 `nth(key_index)`；`ParsedFeatureHash` `parse_structured`/`parse_list_split`/`parse_structured_flat_split` 同样处理 | ~9060 / ~9960 / ~10180 | ~9070 / ~9970 / ~10190 | 保留；这些算子不在前 3 热点中，改动使代码一致且减少分配 |
| E | `StringConcat`/`ConcatHash` 去掉 `collect::<Vec<_>>().join()` 中间 Vec 分配，改为直接 `push_str` 累积；对 `Fv::Str` 直接引用避免 `to_string()` 分配 | ~9070 / ~9970 / ~10190 | ~9070 / ~9970 / ~10190 | 保留；总吞吐持平（算子本身 <0.003s，改动后减少分配但不在热点路径上） |
| F | `JsonExtractList` 添加 7 个边缘 case 测试（bool/number 值、key 缺失、非法 JSON、空字符串值、空数组、转义字符串、no-key 对象数组）。`FlatSplit` `process_batch` 消除不必要的 `list.clone()` 改用 `&[String]` 借用 | 9557.8 / 10156.4 / 10324.7 | 同上 | 已提交；JsonExtractList 使用 serde_json <100 字节 JSON 无性能差异，自定义字节扫描器因分支密集反而不如 serde_json 表驱动解析器；新增测试保留用于回归覆盖 |

`JsonExtractList` 本次尝试用自定义字节扫描器替代 `serde_json::from_str` DOM 分配，但 demo JSON 字符串极短（15-100 字节），serde_json 的 SIMD 表驱动解析已是最优选择；自定义扫描器因密集分支判断和字节级操作未能胜出，已回退。新增的 7 个边缘 case 测试保留。

撤回尝试：`JsonExtractList` 预分配 `pad_len` 并在收够输出后停止遍历。benchmark 未提升（约 4012/4259/4224 rows/s），原因是 `serde_json::from_str` 仍完整解析 JSON，减少后续转换不足以抵消波动；该改动未保留。

撤回尝试：`JsonExtractList` + `ParsedFeatureHash` 换用 `simd-json` 替代 `serde_json::from_str`。由于 demo 数据 JSON 字符串极短（15-100 字节），simd-json 的 SIMD 加速优势无法发挥，且为获取 `&mut str` 需额外 clone 输入字符串，提升 < 1%。该改动未保留。

## 新增/修改文件清单

| 文件 | 说明 |
|------|------|
| `crates/core/Cargo.toml` | 新增 core crate，依赖 serde/petgraph/rhai/libloading/tracing |
| `crates/core/src/lib.rs` | 只暴露 `pub mod feats` |
| `crates/core/src/feats/mod.rs` | 子模块声明 + 导出 `FeatureSpec` |
| `crates/core/src/feats/config.rs` | 新增 `FeatureSpec` struct |
| `crates/core/src/feats/defaults.rs` | 新增 `parse_string_to_fv()` |
| `crates/core/src/feats/executor.rs` | 优化单输出列赋值，避免不必要的结果列 clone |
| `crates/core/src/feats/ops/expression.rs` | `ExpressionOp` 专用 batch 路径，复用 Rhai scope 和变量名 |
| `crates/core/src/feats/ops/feature_hash.rs` | 优化单输入标量/list batch hash 快路径，减少 key 和中间列表分配 |
| `crates/core/src/feats/ops/list_string_parser.rs` | 去掉 `collect::<Vec<&str>>()` 改用 `nth(key_index)`，消除中间 Vec 分配 |
| `crates/core/src/feats/ops/parsed_feature_hash.rs` | 同上，`parse_structured`/`parse_list_split`/`parse_structured_flat_split` 去掉 Collect Vec |
| `crates/core/src/feats/ops/string_concat.rs` | 去掉 `collect::<Vec<_>>().join()` 改用直接 `push_str` 累积，`Fv::Str` 零分配拼接 |
| `crates/core/src/feats/ops/concat_hash.rs` | 同上 |
| `crates/core/src/feats/ops/json_extract_list.rs` | 提取 `extract_values()` 辅助函数；添加 7 个边缘 case 测试（保留 serde_json 实现） |
| `crates/core/src/feats/ops/flat_split.rs` | `process_batch` 消除 `list.clone()`，改用 `&[String]` 借用 |
| `crates/core/src/feats/ops/mod.rs` | 默认 batch fallback 复用行缓冲，减少逐行分配 |
| `crates/core/src/feats/tensor_utils.rs` | 新增：`feature_column_to_vec()` 共享 pooling/padding |
| `Cargo.toml` | 改为 `[workspace]`，members = `crates/core` |
| `src/feats/mod.rs` | 改为 `pub use scale_rec_core::feats::*` + 本地模块 |
| `src/layers/embedding.rs` | 删除 FeatureSpec 定义，改为 `pub use crate::feats::FeatureSpec` |
| `python/rust_feat_bridge/` | 新增 PyO3 crate (feat_engine Python 包) |
| `python/src/train/core/dag.py` | 新增 `use_rust`/`config_path` 参数 |
| `python/src/scale_rec_demo/benchmark_preprocess.py` | Python/Rust 预处理吞吐 benchmark |
| `AGENTS.md` | 更新构建和测试命令 |

## 关键决策

1. **PyO3 而非子进程** — 零拷贝传递 Python 对象，GIL 可释放
2. **返回值 `list[int]` 而非 torch.Tensor** — 避免引入 `tch-rs`，保持 bridge 轻量；代价是 Python 侧仍需构造 tensor
3. **严格 dtype 解析** — 与推理 `json_to_feature_typed()` 一致；坏数据用 `default_val`
4. **Python 侧 str 转换** — 调用 Rust 前 `str(v) if v is not None else None`，避免 PyO3 类型冲突
5. **Fallback** — `import feat_engine` 失败时自动回退 Python 路径
