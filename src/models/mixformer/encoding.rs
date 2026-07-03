//! MixFormer core modules: HeadMixing, SwiGLU FFN, QueryMixer, OutputFusion.
//!
//! Paper §3.3: Each MixFormer block = QueryMixer + CrossAttention + OutputFusion.
//! CrossAttention depends on sequence data; QueryMixer and OutputFusion are
//! implemented here.

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear_no_bias, LayerNorm, Linear, VarBuilder};

/// Parameter-free cross-head information exchange.
///
/// Paper §3.3.1: Reshapes [B, N, D] → [B, N, N, D/N],
/// transposes dims 1↔2, and flattens back to [B, N, D].
fn head_mixing(x: &Tensor, n: usize) -> Result<Tensor> {
    let (b, _n, d) = x.dims3()?;
    let chunk = d / n;
    let x = x.reshape((b, n, n, chunk))?;
    let x = x.permute((0, 2, 1, 3))?;
    x.reshape((b, n, d))
}

/// SwiGLU-activated FFN: SwiGLU(x) = (SiLU(x·W_gate)⊗(x·W_up))·W_down
struct SwiGLUFFN {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SwiGLUFFN {
    /// Create SwiGLU FFN with gate, up, down projections.
    fn new(vb: VarBuilder, d: usize, d_ff: usize) -> Result<Self> {
        let gate = linear_no_bias(d, d_ff, vb.pp("gate"))?;
        let up = linear_no_bias(d, d_ff, vb.pp("up"))?;
        let down = linear_no_bias(d_ff, d, vb.pp("down"))?;
        Ok(Self { gate, up, down })
    }

    /// Forward: SwiGLU(x) = (SiLU(x·W_gate)⊗(x·W_up))·W_down.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let silu = candle_nn::ops::sigmoid(&gate)?.mul(&gate)?;
        self.down.forward(&silu.mul(&up)?)
    }
}

/// Query Mixer: HeadMixing + per-head SwiGLU FFN.
///
/// Paper §3.3.1 Eq. (3)–(4):
///   P = HeadMixing(Norm(X)) + X
///   Q_i = SwiGLUFFN_i(Norm(P_i)) + P_i
pub struct QueryMixer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    head_ffns: Vec<SwiGLUFFN>,
    num_heads: usize,
}

impl QueryMixer {
    /// Create a new Query Mixer.
    pub fn new(vb: VarBuilder, d: usize, d_ff: usize, num_heads: usize) -> Result<Self> {
        let norm1 = layer_norm(d, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(d, 1e-5, vb.pp("norm2"))?;
        let mut head_ffns = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            head_ffns.push(SwiGLUFFN::new(vb.pp(format!("head_ffns.{}", i)), d, d_ff)?);
        }
        Ok(Self { norm1, norm2, head_ffns, num_heads })
    }

    /// Forward: HeadMixing → per-head SwiGLU FFN.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // HeadMixing block
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = head_mixing(&x, self.num_heads)?;
        let x = (residual + &x)?;

        // Per-head FFN
        let residual = &x;
        let x = self.norm2.forward(&x)?;
        let mut outputs = Vec::with_capacity(self.num_heads);
        for i in 0..self.num_heads {
            let head = x.narrow(1, i, 1)?.squeeze(1)?;
            outputs.push(self.head_ffns[i].forward(&head)?.unsqueeze(1)?);
        }
        let x = Tensor::cat(&outputs, 1)?;
        residual + &x
    }
}

/// Output Fusion: per-head SwiGLU FFN + residual.
///
/// Paper §3.3.3 Eq. (9):
///   o_i = SwiGLUFFN_i(Norm(z_i)) + z_i
pub struct OutputFusion {
    norm: LayerNorm,
    head_ffns: Vec<SwiGLUFFN>,
    num_heads: usize,
}

impl OutputFusion {
    /// Create a new Output Fusion layer.
    pub fn new(vb: VarBuilder, d: usize, d_ff: usize, num_heads: usize) -> Result<Self> {
        let norm = layer_norm(d, 1e-5, vb.pp("norm"))?;
        let mut head_ffns = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            head_ffns.push(SwiGLUFFN::new(vb.pp(format!("head_ffns.{}", i)), d, d_ff)?);
        }
        Ok(Self { norm, head_ffns, num_heads })
    }

    /// Forward pass.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.norm.forward(x)?;
        let mut outputs = Vec::with_capacity(self.num_heads);
        for i in 0..self.num_heads {
            let head = x.narrow(1, i, 1)?.squeeze(1)?;
            outputs.push(self.head_ffns[i].forward(&head)?.unsqueeze(1)?);
        }
        let x = Tensor::cat(&outputs, 1)?;
        residual + &x
    }
}

/// One MixFormer block: QueryMixer → OutputFusion.
pub struct MixFormerBlock {
    query_mixer: QueryMixer,
    output_fusion: OutputFusion,
}

impl MixFormerBlock {
    /// Create a new MixFormer block.
    pub fn new(vb: VarBuilder, d: usize, d_ff: usize, num_heads: usize) -> Result<Self> {
        let query_mixer = QueryMixer::new(vb.pp("query_mixer"), d, d_ff, num_heads)?;
        let output_fusion = OutputFusion::new(vb.pp("output_fusion"), d, d_ff, num_heads)?;
        Ok(Self { query_mixer, output_fusion })
    }

    /// Forward pass: QueryMixer → OutputFusion.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.query_mixer.forward(x)?;
        self.output_fusion.forward(&x)
    }
}
