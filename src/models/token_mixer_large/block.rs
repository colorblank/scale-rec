//! TokenMixer-Large Block: Mixing & Reverting + PerTokenSwiGLU.
use crate::models::unimixer::per_token_swiglu::PerTokenSwiGlu;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// TokenMixer-Large block with Mixing & Reverting paradigm.
pub struct TokenMixerLargeBlock {
    pub embed_dim: usize,
    pub num_tokens: usize,
    pub token_dim: usize,
    pub num_heads: usize,
    head_pswiglu: PerTokenSwiGlu,
    token_pswiglu: PerTokenSwiGlu,
}

impl TokenMixerLargeBlock {
    /// Construct a TokenMixer-Large block.
    ///
    /// - `embed_dim`: total embedding dimension (= num_tokens * token_dim)
    /// - `token_dim`: dimension per token
    /// - `num_tokens`: number of tokens
    /// - `num_heads`: number of heads for mixing (must divide embed_dim)
    /// - `hidden_factor`: hidden dimension multiplier for SwiGLU
    /// - `vb`: variable builder
    /// - `down_init_scale`: down-projection init scaling (TokenMixer-Large: 0.01)
    pub fn new(
        embed_dim: usize,
        token_dim: usize,
        num_tokens: usize,
        num_heads: usize,
        hidden_factor: f64,
        vb: VarBuilder,
        down_init_scale: f64,
    ) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            candle_core::bail!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                embed_dim,
                num_heads
            );
        }
        let head_token_dim = num_tokens * token_dim / num_heads;
        let head_pswiglu = PerTokenSwiGlu::new(
            num_heads,
            head_token_dim,
            hidden_factor,
            vb.pp("head_pswiglu"),
            down_init_scale,
        )?;
        let token_pswiglu = PerTokenSwiGlu::new(
            num_tokens,
            token_dim,
            hidden_factor,
            vb.pp("token_pswiglu"),
            down_init_scale,
        )?;
        Ok(Self {
            embed_dim,
            num_tokens,
            token_dim,
            num_heads,
            head_pswiglu,
            token_pswiglu,
        })
    }

    /// Forward: Mixing & Reverting → per-token SwiGLU → residual.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, _) = x.dims2()?;
        let h = self.num_heads;

        let x_2d = x.reshape((batch_size, self.num_tokens, self.token_dim))?;

        // Mixing: split by head, concat across tokens
        let x_heads = x_2d.reshape((batch_size, self.num_tokens, h, self.token_dim / h))?;
        let x_hm = x_heads.permute((2, 0, 1, 3))?;
        let head_input = x_hm
            .reshape((h, batch_size, self.num_tokens * self.token_dim / h))?
            .permute((1, 0, 2))?;
        let head_mixed = self.head_pswiglu.forward(&head_input)?;

        // Reverting: split back by token, concat heads
        let head_mixed_2d = head_mixed.permute((1, 0, 2))?;
        let x_revert = head_mixed_2d
            .reshape((h, batch_size, self.num_tokens, self.token_dim / h))?
            .permute((2, 1, 0, 3))?
            .reshape((batch_size, self.num_tokens, self.token_dim))?;

        let token_mixed = self.token_pswiglu.forward(&x_revert)?;
        let output = (token_mixed.add(&x_2d)?).reshape((batch_size, self.embed_dim))?;
        Ok(output)
    }
}
