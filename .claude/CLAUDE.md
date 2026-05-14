# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

scale-rec is a recommendation system with a **Rust inference engine** (Candle) and a **Python training pipeline** (PyTorch + Polars). Both sides share the same YAML feature config and model architectures. Python trains and exports weights as safetensors; Rust loads them via Candle VarMap for inference.

## Commands

```bash
# Rust
cargo check                     # type check
cargo fmt                       # format all Rust code
cargo test                      # run all tests (20 tests)
cargo test --test model_smoke   # run only integration tests
cargo run                       # run inference example

# Python (run from python/ directory)
uv run python -m train.main --feature-config ../examples/feature_config.yaml --model-config config/model_lr.yaml --data data/train.parquet --epochs 10
uv run pytest tests/ -v         # run all Python tests (13 tests)
uvx ruff check src/train/       # lint (ruff with E402 ignored)
uvx ruff format src/train/      # format
uv add <package>                # add dependency
uv add --dev <package>          # add dev dependency
```

Python code uses uv for package management. No need to activate venv — `uv run` handles it. `ruff` is used (not black/isort/flake8).

## Architecture

### Feature preprocessing (shared between Rust and Python)

Both sides parse the same `examples/feature_config.yaml` which defines:
- **sources**: raw input features with optional `embed` config (vocab_size, embed_dim)
- **operators**: a DAG of 7 operator types (`Bucketing`, `DictMapper`, `StringParser`, `CrossFeature`, `ExpressionOp`, `SequenceOp`, `PluginOp`)

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
