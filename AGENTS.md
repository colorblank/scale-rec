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
# → 输出 examples/feature_config_discover.yaml

# ── 训练 (demo 单文件模式) ──
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/train_discover.py \
  --data python/demo/temp/discover_train_data.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config python/demo/model_discover_esmm.yaml \
  --epochs 30 --batch-size 128 --lr 0.005

# ── 训练 (生产流式模式) ──
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/train_discover.py \
  --user-data data/user_20260331.txt \
  --item-files data/items/20260325.txt,data/items/20260326.txt,...,data/items/20260331.txt \
  --feature-config examples/feature_config_discover.yaml \
  --model-config python/demo/model_discover_esmm.yaml \
  --epochs 10 --batch-size 1024 \
  --no-header --null-markers 'NULL' '\N' \
  --skip-missing-item --eval-samples 2000

# ── 训练 (旧 demo 配置) ──
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/train_all.py \
  --feature-config python/demo/feature_config_demo.yaml \
  --data python/demo/temp/train_data.csv \
  --epochs 50 --batch-size 64

# ── 生成合成数据 ──
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/generate_data.py
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/generate_discover_data.py

# ── PyTorch vs Rust 推理一致性验证 ──
PYTHONPATH=python/src:$PYTHONPATH uv run python python/demo/verify_all.py
```

### train_discover.py 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--data` | str | — | 单文件路径（demo 模式，二选一） |
| `--user-data` | str | — | 用户行为文件（生产模式，二选一） |
| `--item-files` | str | — | 物品文件逗号列表（生产模式，必需） |
| `--feature-config` | str | `examples/feature_config_discover.yaml` | 特征编排配置 |
| `--model-config` | str | `python/demo/model_discover_esmm.yaml` | 模型配置 |
| `--export-path` | str | `python/demo/temp/model.safetensors` | 权重导出路径 |
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

Both sides parse the same `examples/feature_config.yaml` which defines:
- **sources**: raw input features (NO embed — all embedding through operators)
- **operators**: a DAG of 14 operator types

全部 14 个算子：`Bucketing`, `DictMapper`, `StringParser`, `JsonExtractList`, `ListStringParser`, `Split`, `FlatSplit`, `ExpressionOp`, `CrossFeature`, `ListOverlap`, `SequenceOp`, `StringConcat`, `FeatureHash`, `PluginOp`。

**配置原则**：
- 默认使用 FeatureHash（无状态哈希），DictMapper 仅用于低基数枚举
- sources 不配 `embed`，全部 embedding 由 operator 输出 `embed` 字段声明
- DAG 构建时自动校验 source 消费率和输出利用率

`FlowConfig` (Rust: `src/feats/config.rs`, Python: `python/src/train/config.py`) deserializes the YAML. `FeatureDag` (Rust: `src/feats/dag.rs`, Python: `python/src/train/dag.py`) builds the DAG with topological sort and executes single samples via `execute(raw_inputs) -> FeatureResult`.

`FeatureDag.embeddable_features()` returns `[(name, EmbedConfig)]` — these are the only features that need embedding. All models get their feature specs from this method; ModelConfig does NOT duplicate feature definitions.

### Model trait and ModelConfig

All Rust models implement `Model` trait (`src/models/mod.rs`):
```rust
pub trait Model: Send + Sync {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>;
}
```

`ModelConfig` enum is YAML-deserializable with `#[serde(tag = "type")]`. Five variants: `LR`, `DeepFM`, `MMoE`, `ESMM`, `UniMixer`. `build(vb, features, tokenizer)` constructs any model from config. Features come from the DAG, not from ModelConfig itself.

ESMM 当前为 5 任务塔（click/cvr/detail/stock/stay），概率关系：
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

### Rust-specific notes

- `candle-core` 0.10 and `candle-nn` 0.10 — pre-1.0 APIs, some methods differ from latest
- `FeatureTokenizer` uses grouped `Conv1d` with `groups=num_tokens` to project heterogeneous embeddings into uniform token sequences
- `PerTokenSwiGlu` implements a custom einsum via batch matmul because Candle 0.10 lacks native einsum
- `UniMixing` uses Sinkhorn-Knopp iteration (3 iterations) to produce doubly stochastic matrices for token interaction

### Python-specific notes

- `preprocess_batch()` in `FeatureDag` handles the full pipeline: row-by-row DAG execution → tensor stacking → `{name: LongTensor [batch]}` dict
- `main.py` `train_epoch()` loops over Polars DataFrame slices, calls `dag.preprocess_batch()` per slice
- Model config YAML files live in `python/config/` and are minimal (e.g., `type: lr` with no features section)
