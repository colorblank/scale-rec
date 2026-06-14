//! RankMixer block: token mixing + per-token FFN.

use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

/// Dedicated two-layer GELU FFN per token.
pub struct PerTokenFfn {
    up: Vec<Linear>,
    down: Vec<Linear>,
}

impl PerTokenFfn {
    /// Construct per-token FFN modules.
    pub fn new(
        num_tokens: usize,
        token_dim: usize,
        hidden_factor: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_dim = ((token_dim as f64) * hidden_factor).floor().max(1.0) as usize;
        let mut up = Vec::with_capacity(num_tokens);
        let mut down = Vec::with_capacity(num_tokens);
        for token_idx in 0..num_tokens {
            up.push(linear(
                token_dim,
                hidden_dim,
                vb.pp(format!("up.{}", token_idx)),
            )?);
            down.push(linear(
                hidden_dim,
                token_dim,
                vb.pp(format!("down.{}", token_idx)),
            )?);
        }
        Ok(Self { up, down })
    }

    /// Forward over `[batch, num_tokens, token_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(self.up.len());
        for token_idx in 0..self.up.len() {
            let token = x.narrow(1, token_idx, 1)?.squeeze(1)?;
            let hidden = self.up[token_idx].forward(&token)?.gelu()?;
            outputs.push(self.down[token_idx].forward(&hidden)?.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }
}

/// Dense RankMixer block from the paper.
pub struct RankMixerBlock {
    token_dim: usize,
    num_tokens: usize,
    num_heads: usize,
    norm_mixing: LayerNorm,
    pffn: PerTokenFfn,
    norm_ffn: LayerNorm,
}

impl RankMixerBlock {
    /// Construct a RankMixer block.
    pub fn new(
        token_dim: usize,
        num_tokens: usize,
        num_heads: usize,
        hidden_factor: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        if num_heads != num_tokens {
            candle_core::bail!("RankMixer requires num_heads == num_tokens for residual shape");
        }
        if token_dim % num_heads != 0 {
            candle_core::bail!(
                "token_dim ({}) must be divisible by num_heads ({})",
                token_dim,
                num_heads
            );
        }
        let norm_mixing = layer_norm(token_dim, 1e-5, vb.pp("norm_mixing"))?;
        let pffn = PerTokenFfn::new(num_tokens, token_dim, hidden_factor, vb.pp("pffn"))?;
        let norm_ffn = layer_norm(token_dim, 1e-5, vb.pp("norm_ffn"))?;
        Ok(Self {
            token_dim,
            num_tokens,
            num_heads,
            norm_mixing,
            pffn,
            norm_ffn,
        })
    }

    fn token_mixing(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.dim(0)?;
        let head_dim = self.token_dim / self.num_heads;
        x.reshape((batch_size, self.num_tokens, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((batch_size, self.num_heads, self.num_tokens * head_dim))
    }

    /// Forward over `[batch, num_tokens, token_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mixed = self.token_mixing(x)?;
        let s = self.norm_mixing.forward(&mixed.add(x)?)?;
        self.norm_ffn.forward(&self.pffn.forward(&s)?.add(&s)?)
    }
}
