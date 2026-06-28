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
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 uv run maturin develop \
  --manifest-path python/rust_feat_bridge/Cargo.toml --uv
```

## 测试

```bash
# Rust 测试
cargo test         # 56 个测试全部通过
cargo check        # 编译检查

# Python 一致性测试（需要先 build feat_engine）
uv run pytest python/tests/test_rust_pretrain_consistency.py -v
```

一致性测试验证：使用相同输入，Rust `FeatSession` 和 Python `FeatureDag` 输出的 45 个 embeddable features 完全相同。

## 新增/修改文件清单

| 文件 | 说明 |
|------|------|
| `crates/core/Cargo.toml` | 新增 core crate，依赖 serde/petgraph/rhai/libloading/tracing |
| `crates/core/src/lib.rs` | 只暴露 `pub mod feats` |
| `crates/core/src/feats/mod.rs` | 子模块声明 + 导出 `FeatureSpec` |
| `crates/core/src/feats/config.rs` | 新增 `FeatureSpec` struct |
| `crates/core/src/feats/defaults.rs` | 新增 `parse_string_to_fv()` |
| `crates/core/src/feats/tensor_utils.rs` | 新增：`feature_column_to_vec()` 共享 pooling/padding |
| `Cargo.toml` | 改为 `[workspace]`，members = `crates/core` |
| `src/feats/mod.rs` | 改为 `pub use scale_rec_core::feats::*` + 本地模块 |
| `src/layers/embedding.rs` | 删除 FeatureSpec 定义，改为 `pub use crate::feats::FeatureSpec` |
| `python/rust_feat_bridge/` | 新增 PyO3 crate (feat_engine Python 包) |
| `python/src/train/core/dag.py` | 新增 `use_rust`/`config_path` 参数 |
| `AGENTS.md` | 更新构建和测试命令 |

## 关键决策

1. **PyO3 而非子进程** — 零拷贝传递 Python 对象，GIL 可释放
2. **返回值 `list[int]` 而非 torch.Tensor** — 避免引入 `tch-rs`，Python `torch.tensor()` 零开销
3. **严格 dtype 解析** — 与推理 `json_to_feature_typed()` 一致；坏数据用 `default_val`
4. **Python 侧 str 转换** — 调用 Rust 前 `str(v) if v is not None else None`，避免 PyO3 类型冲突
5. **Fallback** — `import feat_engine` 失败时自动回退 Python 路径
