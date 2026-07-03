//! Field-Aware FFN (§3.3) — per-field W₁ and W₂ projections with SiLU activation.
//!
//! FAT-FFN(z_i) = SiLU(z_i · W_1^{(f_i)}) · W_2^{(f_i)}

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

/// Field-Aware Feed-Forward Network.
#[allow(dead_code)]
pub struct FieldAwareFFN {
    d: usize,
    d_ff: usize,
    norm: LayerNorm,
}

impl FieldAwareFFN {
    /// Create a new field-aware FFN layer.
    pub fn new(vb: VarBuilder, d: usize, d_ff: usize) -> Result<Self> {
        let norm = layer_norm(d, 1e-5, vb.pp("norm"))?;
        Ok(Self { d, d_ff, norm })
    }

    /// Forward pass with pre-computed field-specific projections.
    ///
    /// Args:
    ///   x: [batch, num_fields, d]  input.
    ///   w_1: [num_fields, d, d_ff]  field-specific input projections.
    ///   w_2: [num_fields, d_ff, d]  field-specific output projections.
    ///
    /// Returns: [batch, num_fields, d]
    #[allow(non_snake_case)]
    pub fn forward(&self, x: &Tensor, w_1: &Tensor, w_2: &Tensor) -> Result<Tensor> {
        // Pre-LN
        let x = self.norm.forward(x)?;

        // FAT-FFN: SiLU(x · W₁) · W₂
        // SiLU(x) = x * sigmoid(x)  (identical to Swish with β=1)
        let hidden = self.field_matmul(&x, w_1)?;           // [B, F, d_ff]
        let sig = candle_nn::ops::sigmoid(&hidden)?;
        let hidden = hidden.mul(&sig)?;                     // SiLU activation
        let output = self.field_matmul(&hidden, w_2)?;     // [B, F, d]

        Ok(output)
    }

    /// Batched per-field matrix multiply:  out[b,f,:] = x[b,f,:] @ W[f,:,:]
    fn field_matmul(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (_batch, num_fields, _) = x.dims3()?;
        let mut outputs = Vec::with_capacity(num_fields);
        for f in 0..num_fields {
            let x_f = x.narrow(1, f, 1)?.squeeze(1)?;     // [B, d_in]
            let w_f = w.get(f)?.squeeze(0)?;                // [d_in, d_out]
            let out_f = x_f.matmul(&w_f)?;                  // [B, d_out]
            outputs.push(out_f.unsqueeze(1)?);               // [B, 1, d_out]
        }
        Tensor::cat(&outputs, 1)
    }
}
