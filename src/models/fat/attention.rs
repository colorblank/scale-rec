//! Field-Decomposed Attention (§3.2) — field-specific Q/K/V projections
//! and field-pair interaction modulation.

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, VarBuilder};

/// Field-Decomposed Multi-Head Attention layer.
#[allow(dead_code)]
pub struct FieldDecomposedAttention {
    d: usize,
    n_heads: usize,
    d_head: usize,
    num_fields: usize,
    norm: LayerNorm,
    field_bias: Option<Tensor>,
}

impl FieldDecomposedAttention {
    /// Create a new field-decomposed attention layer.
    pub fn new(vb: VarBuilder, d: usize, n_heads: usize, num_fields: usize) -> Result<Self> {
        let norm = layer_norm(d, 1e-5, vb.pp("norm"))?;
        let field_bias = if num_fields > 0 {
            Some(vb.get_with_hints(
                (num_fields, d),
                "field_bias",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
            )?)
        } else {
            None
        };

        Ok(Self {
            d,
            n_heads,
            d_head: d / n_heads,
            num_fields,
            norm,
            field_bias,
        })
    }

    /// Forward pass with pre-computed field-specific projections.
    ///
    /// Args:
    ///   x: [batch, num_fields, d]  input token embeddings.
    ///   w_q: [num_fields, d, d]  field-specific Q projections.
    ///   w_k: [num_fields, d, d]  field-specific K projections.
    ///   w_v: [num_fields, d, d]  field-specific V projections.
    ///   field_pair_w: [num_fields, num_fields]  interaction modulation.
    ///
    /// Returns: [batch, num_fields, d]
    #[allow(non_snake_case)]
    pub fn forward(
        &self,
        x: &Tensor,
        w_q: &Tensor,
        w_k: &Tensor,
        w_v: &Tensor,
        field_pair_w: &Tensor,
    ) -> Result<Tensor> {
        let (batch, num_fields, _d) = x.dims3()?;

        // Pre-LN
        let x = self.norm.forward(x)?;

        // Add field-aware bias:  h_i = e_i + b_{f_i}
        let x = if let Some(ref bias) = self.field_bias {
            x.broadcast_add(bias)?
        } else {
            x
        };

        // Per-field Q/K/V projections
        //   q_i = h_i · W_Q^{(f_i)}  →  Q[b,f,:] = Σ_e x[b,f,e] · W_q[f,e,:]
        // Equivalent to batched matmul:  [B,F,d] × [F,d,d] → [B,F,d]
        let q = self.field_matmul(&x, w_q)?;
        let k = self.field_matmul(&x, w_k)?;
        let v = self.field_matmul(&x, w_v)?;

        // Reshape to multi-head:  [B, F, d] → [B, n_heads, F, d_head]
        let q = q
            .reshape((batch, num_fields, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((batch, num_fields, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((batch, num_fields, self.n_heads, self.d_head))?
            .permute((0, 2, 1, 3))?;

        // Scaled dot-product attention
        // scores [B, n_heads, F, F] = Q @ K^T / sqrt(d_head)
        let scale = (self.d_head as f64).sqrt();
        let scores = q
            .matmul(&k.permute((0, 1, 3, 2))?)?
            .broadcast_div(&Tensor::from_slice(&[scale], (1,), x.device())?)?;

        // Field-pair interaction modulation
        // field_pair_w: [F, F] → broadcast to [1, 1, F, F]
        let fw = field_pair_w.unsqueeze(0)?.unsqueeze(0)?;
        let scores = scores.broadcast_mul(&fw)?;

        // Softmax + weighted sum over values
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v)?; // [B, n_heads, F, d_head]

        // Reshape back: [B, n_heads, F, d_head] → [B, F, d]
        let out = out
            .permute((0, 2, 1, 3))?
            .reshape((batch, num_fields, self.d))?;

        Ok(out)
    }

    /// Batched per-field matrix multiply.
    ///
    /// For each field f, computes:  out[b,f,:] = x[b,f,:] @ W[f,:,:]
    ///
    /// Equivalent to einsum("bfd,fde->bfe", x, W).
    fn field_matmul(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (_batch, num_fields, _d) = x.dims3()?;
        // Iterate over fields (num_fields is typically small, F ~ 10-100)
        let mut outputs = Vec::with_capacity(num_fields);
        for f in 0..num_fields {
            let x_f = x.narrow(1, f, 1)?.squeeze(1)?; // [B, d]
            let w_f = w.get(f)?.squeeze(0)?; // [d, d]
            let out_f = x_f.matmul(&w_f)?; // [B, d]
            outputs.push(out_f.unsqueeze(1)?); // [B, 1, d]
        }
        Tensor::cat(&outputs, 1)
    }
}
