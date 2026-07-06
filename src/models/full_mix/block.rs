//! Full-Mix block: parameterized full token mixing + GLU-improved per-token FFNs.

use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

/// Dedicated GLU-style FFN per token (paper Eq. 4).
///
///   output_t = (GELU(M_t · W₁) ⊙ (M_t · W₂)) · W₃ + M_t · Wᵣ
///
/// where Wᵣ is a learnable residual projection (not merely an identity shortcut).
pub struct PerTokenGluFfn {
    up: Vec<Linear>,
    gate: Vec<Linear>,
    down: Vec<Linear>,
    skip: Vec<Linear>,
}

impl PerTokenGluFfn {
    /// Construct per-token GLU FFN modules.
    pub fn new(
        num_tokens: usize,
        token_dim: usize,
        hidden_factor: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        if hidden_factor <= 0.0 {
            candle_core::bail!("hidden_factor must be > 0");
        }
        let hidden_dim = ((token_dim as f64) * hidden_factor).floor().max(1.0) as usize;
        let mut up = Vec::with_capacity(num_tokens);
        let mut gate = Vec::with_capacity(num_tokens);
        let mut down = Vec::with_capacity(num_tokens);
        let mut skip = Vec::with_capacity(num_tokens);
        for token_idx in 0..num_tokens {
            up.push(linear(
                token_dim,
                hidden_dim,
                vb.pp("up").pp(token_idx.to_string()),
            )?);
            gate.push(linear(
                token_dim,
                hidden_dim,
                vb.pp("gate").pp(token_idx.to_string()),
            )?);
            down.push(linear(
                hidden_dim,
                token_dim,
                vb.pp("down").pp(token_idx.to_string()),
            )?);
            skip.push(linear(
                token_dim,
                token_dim,
                vb.pp("skip").pp(token_idx.to_string()),
            )?);
        }
        Ok(Self {
            up,
            gate,
            down,
            skip,
        })
    }

    /// Forward over `[batch, num_tokens, token_dim]`.
    ///
    /// Paper Eq. 4: Z_t = (GELU(M_t·W₁) ⊙ (M_t·W₂))·W₃ + M_t·Wᵣ
    /// Gate path uses identity activation (no sigmoid).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(self.up.len());
        for token_idx in 0..self.up.len() {
            let token = x.narrow(1, token_idx, 1)?.squeeze(1)?;
            let up = self.up[token_idx].forward(&token)?.gelu()?;
            let gate = self.gate[token_idx].forward(&token)?; // identity activation
            let hidden = up.mul(&gate)?;
            let gated = self.down[token_idx].forward(&hidden)?;
            let residual = self.skip[token_idx].forward(&token)?;
            outputs.push(gated.add(&residual)?.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }
}

/// Full-Mix block with learnable mixing over flattened token representations.
pub struct FullMixBlock {
    token_dim: usize,
    num_tokens: usize,
    full_mixing: Linear,
    norm_mixing: LayerNorm,
    pffn: PerTokenGluFfn,
    norm_ffn: LayerNorm,
}

impl FullMixBlock {
    /// Construct a Full-Mix block.
    pub fn new(
        token_dim: usize,
        num_tokens: usize,
        hidden_factor: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        if token_dim == 0 {
            candle_core::bail!("token_dim must be > 0");
        }
        if num_tokens == 0 {
            candle_core::bail!("num_tokens must be > 0");
        }
        if hidden_factor <= 0.0 {
            candle_core::bail!("hidden_factor must be > 0");
        }
        let flat_dim = token_dim * num_tokens;
        Ok(Self {
            token_dim,
            num_tokens,
            full_mixing: linear(flat_dim, flat_dim, vb.pp("full_mixing"))?,
            norm_mixing: layer_norm(token_dim, 1e-5, vb.pp("norm_mixing"))?,
            pffn: PerTokenGluFfn::new(num_tokens, token_dim, hidden_factor, vb.pp("pffn"))?,
            norm_ffn: layer_norm(token_dim, 1e-5, vb.pp("norm_ffn"))?,
        })
    }

    /// Forward over `[batch, num_tokens, token_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let flat_dim = self.num_tokens * self.token_dim;
        let mixed = self
            .full_mixing
            .forward(&x.reshape((batch_size, flat_dim))?)?
            .reshape((batch_size, self.num_tokens, self.token_dim))?;
        let s = self.norm_mixing.forward(&mixed.add(x)?)?;
        self.norm_ffn.forward(&self.pffn.forward(&s)?.add(&s)?)
    }
}
