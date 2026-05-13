use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use super::per_token_swiglu::PerTokenSwiGlu;
use super::siamese_norm::SiameseNorm;
use super::unimixing::UniMixing;
use super::unimixing_lite::UniMixingLite;

pub enum UniMixingLayer { Standard(UniMixing), Lite(UniMixingLite) }

pub enum BlockOutput { Siamese((), Tensor, Tensor), Standard(Tensor) }

pub struct UniMixerBlock {
    pub embed_dim: usize,
    pub token_dim: usize,
    pub num_tokens: usize,
    unimixing: UniMixingLayer,
    pswiglu: PerTokenSwiGlu,
    siamese_norm: SiameseNorm,
}

impl UniMixerBlock {
    pub fn new(embed_dim: usize, block_size: usize, token_dim: usize, num_tokens: usize, use_lite: bool, hidden_factor: f64, num_basis: usize, rank: usize, vb: VarBuilder) -> Result<Self> {
        let unimixing = if use_lite {
            let lite = UniMixingLite::new(embed_dim, block_size, num_basis, rank, vb.pp("unimixing_lite"))?;
            UniMixingLayer::Lite(lite)
        } else {
            let standard = UniMixing::new(embed_dim, block_size, vb.pp("unimixing"))?;
            UniMixingLayer::Standard(standard)
        };
        let pswiglu = PerTokenSwiGlu::new(num_tokens, token_dim, hidden_factor, vb.pp("pswiglu"))?;
        let siamese_norm = SiameseNorm::new(embed_dim, 1e-5, vb.pp("siamese_norm"))?;
        Ok(Self { embed_dim, token_dim, num_tokens, unimixing, pswiglu, siamese_norm })
    }

    fn apply_unimixing(&self, x: &Tensor, temperature: f64) -> Result<Tensor> {
        match &self.unimixing {
            UniMixingLayer::Standard(layer) => layer.forward(x, temperature),
            UniMixingLayer::Lite(layer) => layer.forward(x, temperature),
        }
    }

    pub fn forward(&self, x: &Tensor, temperature: f64, x_bar_opt: Option<&Tensor>, y_bar_opt: Option<&Tensor>, use_siamese: bool) -> Result<BlockOutput> {
        let (batch_size, _) = x.dims2()?;
        if use_siamese {
            let x_bar = x_bar_opt.unwrap();
            let y_bar = y_bar_opt.unwrap();
            let y_bar_normalized = self.siamese_norm.forward_rmsnorm(y_bar)?;
            let mixed_input = x_bar.broadcast_add(&y_bar_normalized)?;
            let mixed = self.apply_unimixing(&mixed_input, temperature)?;
            let mixed_tokens = mixed.reshape((batch_size, self.num_tokens, self.token_dim))?;
            let pswiglu_output = self.pswiglu.forward(&mixed_tokens)?;
            let pswiglu_output_flat = pswiglu_output.reshape((batch_size, self.embed_dim))?;
            let block_output = pswiglu_output_flat.broadcast_add(&mixed)?;
            let x_bar_added = x_bar.broadcast_add(&block_output)?;
            let x_bar_new = self.siamese_norm.forward_rmsnorm(&x_bar_added)?;
            let y_bar_new = y_bar.broadcast_add(&block_output)?;
            Ok(BlockOutput::Siamese((), x_bar_new, y_bar_new))
        } else {
            let mixed = self.apply_unimixing(x, temperature)?;
            let mixed_tokens = mixed.reshape((batch_size, self.num_tokens, self.token_dim))?;
            let pswiglu_output = self.pswiglu.forward(&mixed_tokens)?;
            let pswiglu_output_flat = pswiglu_output.reshape((batch_size, self.embed_dim))?;
            let output = mixed.broadcast_add(&pswiglu_output_flat)?;
            Ok(BlockOutput::Standard(output))
        }
    }
}
