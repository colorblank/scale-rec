# scale-rec

Rust + Python recommendation system with feature preprocessing and model training.

## Structure

- `src/` — Rust inference engine (Candle)
  - `feats/` — Feature preprocessing DAG
  - `layers/` — Reusable neural network layers
  - `models/` — LR, DeepFM, MMoE, ESMM, UniMixer
- `python/` — Python training pipeline (PyTorch)
- `examples/` — YAML config files
