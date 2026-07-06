//! OneRank: Unified Transformer-Native Ranking Architecture — KDD 2026.
//!
//! Paper: arXiv:2606.16838
//!
//! Architecture:
//!   1. FeatureEmbeddings → per-field tokens [B, F, d]
//!   2. Task token injection + structured masking
//!   3. L × OneRankBlock (Pre-LN MHSA + Pre-LN FFN)
//!   4. Task extraction + feature pooling → SD projection
//!   5. Cross-task attention (configurable mask)
//!   6. Dynamic matching scoring: s_k = z_k^T · r_k

pub mod encoding;
pub mod model;
pub mod prediction;
