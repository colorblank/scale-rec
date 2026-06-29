# Rust 训练预处理引擎

## 目标

将训练特征预处理（DAG 算子执行 + tensor 转换）下沉到 Rust，使训练与推理共用完全相同的算子代码和 pooling 逻辑。CSV/TSV 读取仍由 pandas 完成。

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

## 数据流

```
训练 (改造后):
  TSV → pandas → {col: [raw_str, ...]}
    → FeatSession.preprocess_batch(columns)
      ├── parse_string_to_fv()  按 dtype 严格解析
      ├── execute_plan()        Rust ExecutionPlan (与推理相同)
      └── feature_column_to_vec()  pooling/padding (与推理相同)
    → {name: list[int]} → torch.tensor() → model.forward()

推理 (不变):
  JSON → rows_to_columns()
    → executor.execute_plan()         ← 同一份 plan 代码
    → feature_column_to_tensor()       ← 同一份 pooling 逻辑
    → Candle Tensor → model.forward()
```

## FeatSession

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

## Python 集成

`dag.py:FeatureDag.__init__` 新增参数 `use_rust` / `config_path`：

```python
dag = FeatureDag(flow_config,
    use_rust=True,
    config_path="examples/shared/feature_config_demo.yaml")
tensors = dag.preprocess_batch(test_data)  # 自动走 Rust 路径
```

- Python DAG 始终构建（metadata 被模型构建、CLI export、EmbeddingBucketTracker 依赖）
- 仅在 `preprocess_batch` 热路径上替换为 Rust
- 若 `import feat_engine` 失败，静默回退到 Python 实现

## 构建

```bash
# 编译 feat_engine 扩展（Python venv 内）
uv run maturin develop \
  --manifest-path python/rust_feat_bridge/Cargo.toml --uv
```

训练入口通过显式参数启用：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --use-rust-preprocess \
  --require-rust-preprocess
```

`--use-rust-preprocess` 启用 Rust 预处理；`--require-rust-preprocess` 会在 `feat_engine` 未构建或初始化失败时直接终止训练，而不是回退 Python 路径。

## 测试

```bash
# Rust 测试
cargo test         # 146 个测试全部通过（67 core + 38 lib + 41 integration）
cargo check        # 编译检查

# Python 一致性测试（需要先 build feat_engine）
uv run pytest python/tests/test_rust_pretrain_consistency.py -v
```

一致性测试验证：使用相同输入，Rust `FeatSession` 和 Python `FeatureDag` 输出的 45 个 embeddable features 完全相同。

## Benchmark

预处理吞吐基准脚本会模拟训练热路径：pandas batch 切片、列转 `list`、`FeatureDag.preprocess_batch()`，以及 Python 侧 `torch.tensor()` 构造。

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

输出指标：

- `prepare_s`：pandas batch 切片后构造 `dict[str, list]` 的耗时。
- `preprocess_s`：`FeatureDag.preprocess_batch()` 耗时，包含 Rust/Python DAG、PyO3 边界、返回值转 tensor。
- `total_s` / `rows/s`：端到端预处理吞吐，用于比较后续优化。
- `--profile`：额外拆分 Python execute/tensor 或 Rust parse/execute/extract，并输出 top operator types / operators。

当前 demo 基线（2000 行，Windows dev build）显示 Rust 路径主要耗时在 `rust_execute_s`；小样本下热点集中在 `FeatureHash`、`StringParser`、`JsonExtractList`、`ListOverlap`。

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
