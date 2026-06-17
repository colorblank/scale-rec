//! UniMixerBlock：单层交互块，组合 UniMixing + SwiGLU + SiameseNorm。
use super::per_token_swiglu::PerTokenSwiGlu;
use super::profile;
use super::siamese_norm::SiameseNorm;
use super::unimixing::UniMixing;
use super::unimixing_lite::UniMixingLite;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// 交互层变体：标准版或轻量版。
pub enum UniMixingLayer {
    /// 标准 UniMixing 层。
    Standard(UniMixing),
    /// 轻量 UniMixing-Lite 层。
    Lite(UniMixingLite),
}

/// 块输出变体：标准流或 Siamese 双流。
pub enum BlockOutput {
    /// Siamese 双流输出。
    Siamese(Tensor, Tensor),
    /// 标准单流输出。
    Standard(Tensor),
}

/// 单层交互块：UniMixing + SwiGLU + SiameseNorm。
pub struct UniMixerBlock {
    /// token 序列展平后的总维度。
    pub embed_dim: usize,
    /// 单个 token 的维度。
    pub token_dim: usize,
    /// token 数量。
    pub num_tokens: usize,
    unimixing: UniMixingLayer,
    pswiglu: PerTokenSwiGlu,
    siamese_norm: SiameseNorm,
}

impl UniMixerBlock {
    /// 构造 UniMixerBlock。
    pub fn new(
        embed_dim: usize,
        block_size: usize,
        token_dim: usize,
        num_tokens: usize,
        use_lite: bool,
        hidden_factor: f64,
        num_basis: usize,
        rank: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let unimixing = if use_lite {
            let lite = UniMixingLite::new(
                embed_dim,
                block_size,
                num_basis,
                rank,
                vb.pp("unimixing_lite"),
            )?;
            UniMixingLayer::Lite(lite)
        } else {
            let standard = UniMixing::new(embed_dim, block_size, vb.pp("unimixing"))?;
            UniMixingLayer::Standard(standard)
        };
        let pswiglu =
            PerTokenSwiGlu::new(num_tokens, token_dim, hidden_factor, vb.pp("pswiglu"), 1.0)?;
        let siamese_norm = SiameseNorm::new(embed_dim, 1e-5, vb.pp("siamese_norm"))?;
        Ok(Self {
            embed_dim,
            token_dim,
            num_tokens,
            unimixing,
            pswiglu,
            siamese_norm,
        })
    }

    fn apply_unimixing(&self, x: &Tensor, temperature: f64) -> Result<Tensor> {
        match &self.unimixing {
            UniMixingLayer::Standard(layer) => layer.forward(x, temperature),
            UniMixingLayer::Lite(layer) => layer.forward(x, temperature),
        }
    }

    /// 预热内部缓存。
    pub fn warmup(&self, temperature: f64) -> Result<()> {
        match &self.unimixing {
            UniMixingLayer::Standard(layer) => layer.warmup(temperature)?,
            UniMixingLayer::Lite(layer) => layer.warmup(temperature)?,
        }
        self.pswiglu.warmup()
    }

    /// 前向传播。
    pub fn forward(
        &self,
        x: &Tensor,
        temperature: f64,
        x_bar_opt: Option<&Tensor>,
        y_bar_opt: Option<&Tensor>,
        use_siamese: bool,
    ) -> Result<BlockOutput> {
        let (batch_size, _) = x.dims2()?;
        if use_siamese {
            let x_bar = x_bar_opt.unwrap();
            let y_bar = y_bar_opt.unwrap();
            let norm_timer = profile::start();
            let y_bar_normalized = self.siamese_norm.forward_rmsnorm(y_bar)?;
            profile::log("block.siamese.y_bar_rmsnorm", norm_timer);
            let mix_input_timer = profile::start();
            let mixed_input = x_bar.broadcast_add(&y_bar_normalized)?;
            profile::log("block.siamese.mix_input_add", mix_input_timer);
            let unimixing_timer = profile::start();
            let mixed = self.apply_unimixing(&mixed_input, temperature)?;
            profile::log("block.unimixing", unimixing_timer);
            let reshape_timer = profile::start();
            let mixed_tokens = mixed.reshape((batch_size, self.num_tokens, self.token_dim))?;
            profile::log("block.mixed_reshape_tokens", reshape_timer);
            let pswiglu_timer = profile::start();
            let pswiglu_output = self.pswiglu.forward(&mixed_tokens)?;
            profile::log("block.pswiglu", pswiglu_timer);
            let output_timer = profile::start();
            let pswiglu_output_flat = pswiglu_output.reshape((batch_size, self.embed_dim))?;
            let block_output = pswiglu_output_flat.broadcast_add(&mixed)?;
            profile::log("block.output_residual", output_timer);
            let stream_timer = profile::start();
            let x_bar_added = x_bar.broadcast_add(&block_output)?;
            let x_bar_new = self.siamese_norm.forward_rmsnorm(&x_bar_added)?;
            let y_bar_new = y_bar.broadcast_add(&block_output)?;
            profile::log("block.siamese.stream_update", stream_timer);
            Ok(BlockOutput::Siamese(x_bar_new, y_bar_new))
        } else {
            let unimixing_timer = profile::start();
            let mixed = self.apply_unimixing(x, temperature)?;
            profile::log("block.unimixing", unimixing_timer);
            let reshape_timer = profile::start();
            let mixed_tokens = mixed.reshape((batch_size, self.num_tokens, self.token_dim))?;
            profile::log("block.mixed_reshape_tokens", reshape_timer);
            let pswiglu_timer = profile::start();
            let pswiglu_output = self.pswiglu.forward(&mixed_tokens)?;
            profile::log("block.pswiglu", pswiglu_timer);
            let output_timer = profile::start();
            let pswiglu_output_flat = pswiglu_output.reshape((batch_size, self.embed_dim))?;
            let output = mixed.broadcast_add(&pswiglu_output_flat)?;
            profile::log("block.output_residual", output_timer);
            Ok(BlockOutput::Standard(output))
        }
    }
}
