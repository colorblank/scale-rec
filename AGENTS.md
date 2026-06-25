# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

scale-rec is a recommendation system with a **Rust inference engine** (Candle) and a **Python training pipeline** (PyTorch + Polars). Both sides share the same YAML feature config and model architectures. Python trains and exports weights as safetensors; Rust loads them via Candle VarMap for inference.

## Commands

所有命令从仓库根目录执行。

### Rust

```bash
cargo check                        # type check
cargo fmt                          # format all Rust code
cargo test                         # run all tests (38 tests)
cargo test --test model_smoke      # run only integration tests
cargo test feats::ops::feature_hash  # run specific module tests
cargo doc --no-deps                # build docs (warn on missing docs)
cargo run                          # run inference example
```

### Python

Python 代码统一使用 uv + ruff，不需要手动激活 venv。所有命令从仓库根目录执行。

```bash
# ── 测试与检查 ──
PYTHONPATH=python/src:$PYTHONPATH uv run pytest python/tests/ -v
uvx ruff check python/src/         # lint
uvx ruff format python/src/        # format
uv add <package>                    # add dependency
uv add --dev <package>              # add dev dependency

# ── 生成特征配置文件 ──
PYTHONPATH=python/src:$PYTHONPATH uv run python examples/gen_discover_config.py
# → 输出 examples/shared/feature_config_discover.yaml

# ── 训练 (demo 单文件模式) ──
PYTHONPATH=python/src:$PYTHONPATH uv run python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400

# ── 训练 (生产流式模式) ──
PYTHONPATH=python/src:$PYTHONPATH uv run python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 --end-date 20260331 \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 1024 --no-header

# ── 生成合成数据 ──
PYTHONPATH=python/src:$PYTHONPATH uv run python -m scale_rec_demo.generate_data
PYTHONPATH=python/src:$PYTHONPATH uv run python -m scale_rec_demo.generate_discover_data

# ── PyTorch vs Rust 推理一致性验证 ──
PYTHONPATH=python/src:$PYTHONPATH uv run python -m scale_rec_demo.verify_all

```

### train_discover.py 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--data` | str | — | 单文件路径（demo 模式，二选一） |
| `--user-data` | str | — | 用户行为文件（生产模式，二选一） |
| `--item-files` | str | — | 物品文件逗号列表（生产模式，必需） |
| `--feature-config` | str | `examples/shared/feature_config_discover.yaml` | 特征编排配置 |
| `--model-config` | str | — | 模型配置（必填） |
| `--export-path` | str | run 目录 `serving/model.safetensors` | 权重导出路径 |
| `--epochs` | int | 30 | 训练轮数 |
| `--batch-size` | int | 64 | 批次大小 |
| `--lr` | float | 0.005 | 学习率 |
| `--weight-decay` | float | 1e-4 | 权重衰减 |
| `--no-header` | flag | — | TSV 不含 header 行 |
| `--null-markers` | str[] | NULL \N null None "" | NULL 字符串标记 |
| `--separator` | str | `\t` | 字段分隔符 |
| `--skip-missing-item` | flag | — | 跳过 item_id 不在索引中的行 |
| `--eval-samples` | int | 2000 | 评估样本数（生产模式截取） |

## Architecture

### Feature preprocessing (shared between Rust and Python)

Both sides parse the same `examples/feature_config_discover.yaml` which defines:
- **sources**: raw input features (NO embed — all embedding through operators)
- **operators**: a DAG of 17 operator types

全部 17 个算子：`Bucketing`, `DictMapper`, `StringParser`, `JsonExtractList`, `ListStringParser`, `Split`, `FlatSplit`, `ExpressionOp`, `Log1p`, `CrossFeature`, `ListOverlap`, `SequenceOp`, `StringConcat`, `FeatureHash`, `PluginOp`, `ParsedFeatureHash`, `ConcatHash`。

**配置原则**：
- 默认使用 FeatureHash（无状态哈希），DictMapper 仅用于低基数枚举
- sources 不配 `embed`，全部 embedding 由 operator 输出 `embed` 字段声明
- DAG 构建时自动校验 source 消费率 and 输出利用率

`FlowConfig` (Rust: `src/feats/config.rs`, Python: `python/src/train/core/config.py`) deserializes the YAML. `FeatureDag` (Rust: `src/feats/dag.rs`, Python: `python/src/train/core/dag.py`) builds the DAG with topological sort and executes single samples via `execute(raw_inputs) -> FeatureResult`.

### Operator registration

Both Rust and Python use a **registry pattern** instead of a central match statement:

| Language | Registry | Factory |
|---|---|---|
| Rust | `src/feats/ops/registry.rs` — `OP_REGISTRY: LazyLock<HashMap<&str, OpFactory>>` | Each operator exports `pub fn create(params) -> Result<Box<dyn CustomOp>>` |
| Python | `python/src/train/ops/__init__.py` — `OP_REGISTRY: dict[str, type]` | Each operator has `@classmethod from_config(params) -> Self` decorated with `@register_op("OpType")` |

To add a new operator:
1. **Rust**: implement `CustomOp` in `src/feats/ops/<name>.rs`, export `pub fn create()`, register in `registry.rs`
2. **Python**: implement class in `python/src/train/ops/<name>.py`, add `@register_op` + `from_config`
3. **No changes** needed in `dag.rs` or `dag.py`.

`FeatureDag.embeddable_features()` returns `[(name, EmbedConfig)]` — these are the only features that need embedding. All models get their feature specs from this method; ModelConfig does NOT duplicate feature definitions.

### Model trait and ModelConfig

All Rust models implement `Model` trait (`src/models/mod.rs`):
```rust
pub trait Model: Send + Sync {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>;
}
```

`ModelConfig` parses `type` plus flattened YAML params and dispatches through the model registry.
Eight model types are registered: `lr`, `deepfm`, `mmoe`, `esmm`, `gdcn_esmm`, `unimixer`,
`token_mixer_large`, `rankmixer`. `build(vb, features, tokenizer)` constructs any model from config.
Features come from the DAG, not from ModelConfig itself.

All registered models support `output_contract.version: 1`. Shared-backbone models expose a
`shared` representation to `OutputHead`; MMoE exposes one named representation for each unique
`graph.towers[].input`. Legacy `tasks/task_config/label_col_map/metrics` remain compatible but
cannot be mixed with `output_contract`.

ESMM/GDCNESMM 当前为 5 任务塔（click/cvr/detail/stock/stay），概率关系：
- P(detail) = σ(click)·σ(detail), P(stock) = σ(click)·σ(stock)
- P(cvr) = σ(click)·σ(cvr), P(stay) = σ(detail)·σ(stay)

Python mirrors this in `python/src/train/models/__init__.py` with a `ModelConfig` dataclass and `build(features, tokenizer=None)` method.

### Weight export: critical naming alignment

Python model `state_dict` keys MUST match Candle `VarBuilder::pp()` paths exactly so that safetensors can be loaded directly. The naming rules:

| Python `nn.Module` pattern | Candle path | Example key |
|---|---|---|
| `setattr(self, f"emb_{name}", nn.Embedding)` | `vb.pp(format!("emb_{}", name))` | `embeddings.emb_user_id.weight` |
| `self.hidden = nn.ModuleDict({"0": Linear, "1": Linear})` | `vb.pp("hidden.0")` | `hidden.0.weight` |
| `self.output = nn.Linear(...)` | `vb.pp("output")` | `output.weight` |
| `self.output = nn.ModuleDict({str(n): Linear})` (TaskTower) | `vb.pp(format!("output.{}", n))` | `output.1.weight` |
| `setattr(self, f"expert_{e}", Mlp(...))` | `vb.pp(format!("expert_{}", e))` | `expert_0.hidden.0.weight` |
| `nn.Parameter(torch.zeros(1))` | `vb.get_with_hints((1,), ..., Const(0.0))` | `global_bias` |

When adding new layers or models, verify naming by checking `print_state_dict_keys(model)` (from `export.py`) against the corresponding Rust `VarBuilder::pp()` paths.

### Embedding bucket tracking and serving export

- `EmbeddingBucketTracker` 在所有实际执行反向传播的训练 batch 上累计完整 bucket hit count；没有 loss 的 batch 不计入。
- tracker 状态保存在 `.resume.pt` 中，断点续训必须恢复并继续累计。
- run 目录输出 `embedding_bucket_report.yaml`，serving manifest 通过 `embedding_bucket_report_file` 引用该报告。
- `DictMapper`、`FeatureHash`、`ParsedFeatureHash`、`ConcatHash` 的零命中 row 只在最终 serving safetensors 中替换为活跃 row 均值；checkpoint 权重禁止修改。
- `DictMapper.default_idx` 没有命中时同样使用活跃 row 均值。整张表没有任何活跃 bucket 时必须拒绝发布。
- Python/Rust 推理一致性比较必须按 `OutputKind` 使用 serving 语义：`binary_logit` 先 sigmoid，`probability/regression/score` 保持原值。

### Rust-specific notes

- `candle-core` 0.10 and `candle-nn` 0.10 — pre-1.0 APIs, some methods differ from latest
- `FeatureTokenizer` uses grouped `Conv1d` with `groups=num_tokens` to project heterogeneous embeddings into uniform token sequences
- `PerTokenSwiGlu` implements a custom einsum via batch matmul because Candle 0.10 lacks native einsum
- `UniMixing` uses Sinkhorn-Knopp iteration (3 iterations) to produce doubly stochastic matrices for token interaction

### Python-specific notes

- `preprocess_batch()` in `FeatureDag` handles the full pipeline: row-by-row DAG execution → tensor stacking → `{name: LongTensor [batch]}` dict
- `main.py` `train_epoch()` loops over Polars DataFrame slices, calls `dag.preprocess_batch()` per slice
- Model config YAML files live in `examples/models/`; all examples use native `output_contract`
  and never duplicate feature definitions.
