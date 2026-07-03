//! OneRank Transformer block with structured attention masking.
//!
//! Paper §2.1–§2.2: Task-specific token injection with mutual invisibility.

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

/// One Transformer block: Pre-LN MHSA + Pre-LN FFN with residual.
pub struct OneRankBlock {
    norm1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    norm2: LayerNorm,
    ffn_1: Linear,
    ffn_2: Linear,
    #[allow(dead_code)]
    d: usize,
    #[allow(dead_code)]
    n_heads: usize,
    d_head: usize,
}

impl OneRankBlock {
    /// Create a new OneRank block.
    pub fn new(vb: VarBuilder, d: usize, d_ff: usize, n_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(d, 1e-5, vb.pp("norm1"))?;
        let q_proj = linear(d, d, vb.pp("attn.q_proj"))?;
        let k_proj = linear(d, d, vb.pp("attn.k_proj"))?;
        let v_proj = linear(d, d, vb.pp("attn.v_proj"))?;
        let o_proj = linear(d, d, vb.pp("attn.o_proj"))?;
        let norm2 = layer_norm(d, 1e-5, vb.pp("norm2"))?;
        let ffn_1 = linear(d, d_ff, vb.pp("ffn.0"))?;
        let ffn_2 = linear(d_ff, d, vb.pp("ffn.2"))?;
        Ok(Self {
            norm1,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            norm2,
            ffn_1,
            ffn_2,
            d,
            n_heads,
            d_head: d / n_heads,
        })
    }

    /// Forward with pre-built attention bias mask.
    ///
    /// Args:
    ///   x: [B, N, d]
    ///   mask: [N, N] where 0.0 = allowed, -inf = masked
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, n, _d) = x.dims3()?;

        // Pre-LN MHSA
        let residual = x;
        let x = self.norm1.forward(x)?;
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;

        // Reshape to multi-head: [B, n_heads, N, d_head]
        let q = q.reshape((b, n, self.n_heads, self.d_head))?.permute((0, 2, 1, 3))?;
        let k = k.reshape((b, n, self.n_heads, self.d_head))?.permute((0, 2, 1, 3))?;
        let v = v.reshape((b, n, self.n_heads, self.d_head))?.permute((0, 2, 1, 3))?;

        // Scaled dot-product: scores [B, n_heads, N, N]
        let scale = (self.d_head as f64).sqrt();
        let mut scores = q.matmul(&k.permute((0, 1, 3, 2))?)?;
        scores = scores.broadcast_div(&Tensor::from_slice(&[scale], (1,), x.device())?)?;

        // Apply mask: broadcast [N, N] → [1, 1, N, N]
        scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;

        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v)?;
        let x = out.permute((0, 2, 1, 3))?.reshape((b, n, self.d))?;
        let x = self.o_proj.forward(&x)?;
        let x = (residual + x)?;

        // Pre-LN FFN (GELU)
        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let x = self.ffn_1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.ffn_2.forward(&x)?;
        let x = (residual + x)?;

        Ok(x)
    }
}

/// Build structured attention mask for OneRank.
///
/// Shape [N, N] where N = F + K:
///   - Features ↔ Features: 0.0 (allowed)
///   - Features → Task tokens: -inf (masked)
///   - Task tokens → Features: 0.0 (allowed)
///   - Task token k → Task token k: 0.0 (self)
///   - Task token k → Task token j (j≠k): -inf (mutual invisibility)
pub fn build_attention_mask(
    num_fields: usize,
    num_tasks: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let n = num_fields + num_tasks;
    let mut data = vec![-f32::INFINITY; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            if i < num_fields && j < num_fields {
                // Features ↔ Features
                data[idx] = 0.0;
            } else if i >= num_fields && j < num_fields {
                // Task tokens → Features
                data[idx] = 0.0;
            } else if i >= num_fields && j == i {
                // Task token self
                data[idx] = 0.0;
            }
        }
    }
    Tensor::from_slice(&data, (n, n), device)
}
