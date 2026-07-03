//! OneRank prediction head: cross-task attention + matching scoring.
//!
//! Paper §2.4: Configurable cross-task masking and dynamic matching scoring.

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

/// Cross-task self-attention with configurable mask.
pub struct CrossTaskAttention {
    d: usize,
    #[allow(dead_code)]
    num_tasks: usize,
    norm1: LayerNorm,
    norm2: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    ffn_1: Linear,
    ffn_2: Linear,
    /// [K, K] cross-task mask: 0.0 = allowed, -inf = masked.
    cross_mask: Tensor,
}

impl CrossTaskAttention {
    /// Create cross-task attention with the given mask type.
    pub fn new(vb: VarBuilder, d: usize, num_tasks: usize, mask_type: &str) -> Result<Self> {
        let norm1 = layer_norm(d, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(d, 1e-5, vb.pp("norm2"))?;
        let q_proj = linear(d, d, vb.pp("q_proj"))?;
        let k_proj = linear(d, d, vb.pp("k_proj"))?;
        let v_proj = linear(d, d, vb.pp("v_proj"))?;
        let ffn_1 = linear(d, d * 4, vb.pp("ffn.0"))?;
        let ffn_2 = linear(d * 4, d, vb.pp("ffn.2"))?;

        let cross_mask = build_cross_task_mask(num_tasks, mask_type, vb.device())?;

        Ok(Self {
            d,
            num_tasks,
            norm1,
            norm2,
            q_proj,
            k_proj,
            v_proj,
            ffn_1,
            ffn_2,
            cross_mask,
        })
    }

    /// Forward: cross-task attention + residual + FFN.
    ///
    /// Args:
    ///   x: [B, K, d] task representations.
    ///
    /// Returns: [B, K, d]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _k, _d) = x.dims3()?;

        let residual = x;
        let x = self.norm1.forward(x)?;
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;

        let scale = (self.d as f64).sqrt();
        let mut scores = q.matmul(&k.permute((0, 2, 1))?)?;
        scores = scores.broadcast_div(&Tensor::from_slice(&[scale], (1,), x.device())?)?;
        scores = scores.broadcast_add(&self.cross_mask.unsqueeze(0)?)?;

        let attn = candle_nn::ops::softmax(&scores, 2)?;
        let x = attn.matmul(&v)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let x = self.ffn_1.forward(&x)?;
        let x = x.gelu()?;
        let x = self.ffn_2.forward(&x)?;
        let x = (residual + x)?;

        Ok(x)
    }
}

/// Build cross-task attention mask.
fn build_cross_task_mask(
    num_tasks: usize,
    mask_type: &str,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let n = num_tasks;
    let mut data = vec![-f32::INFINITY; n * n];
    match mask_type {
        "parallel" => {
            for i in 0..n {
                data[i * n + i] = 0.0;
            }
        }
        "null" => {
            data = vec![0.0; n * n];
        }
        "cascade" => {
            for i in 0..n {
                for j in 0..=i {
                    data[i * n + j] = 0.0;
                }
            }
        }
        other => candle_core::bail!("Unknown cross_task_mask: {other}"),
    }
    Tensor::from_slice(&data, (n, n), device)
}

/// Dynamic matching scoring: s_k = sum(z_k * r_k) per task.
///
/// Paper §2.4 Eq. (12): Inner product between task-aware global context
/// and context-conditioned candidate embeddings.
pub fn matching_score(global_repr: &Tensor, task_repr: &Tensor) -> Result<Tensor> {
    let product = global_repr.mul(task_repr)?;
    product.sum(2)
}
