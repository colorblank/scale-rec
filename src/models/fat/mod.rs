//! Field-Aware Transformer (FAT) — KDD 2026.
//!
//! Paper: From Scaling to Structured Expressivity: Rethinking Transformers
//!        for CTR Prediction.  arXiv:2511.12081
//!
//! Architecture:
//!   FeatureEmbeddings → field-aware bias + L× FAT blocks
//!     (Field-Decomposed Attention §3.2, Field-Aware FFN §3.3)
//!     → sum pooling → output head
//!
//! Field-specific projections are synthesised by Basis-Composed Hypernetwork (§3.4),
//! pre-computed once at model build time for inference efficiency.

pub mod attention;
pub mod ffn;
pub mod hypernetwork;
pub mod model;
