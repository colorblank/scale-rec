//! MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders — KDD 2026.
//!
//! Paper: arXiv:2602.14110
//!
//! Architecture:
//!   1. FeatureEmbeddings → concat → project → [B, N, D] (N query heads)
//!   2. L × MixFormerBlock (QueryMixer → OutputFusion)
//!   3. Mean pool heads → [B, D] → OutputHead

pub mod encoding;
pub mod model;
