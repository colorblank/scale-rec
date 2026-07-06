use candle_core::{Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};

pub(super) struct GateNu {
    pub(super) fc1: candle_nn::Linear,
    pub(super) fc2: candle_nn::Linear,
    pub(super) gamma: f64,
}

impl GateNu {
    pub(super) fn new(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            fc1: linear(input_dim, hidden_dim, vb.pp("fc1"))?,
            fc2: linear(hidden_dim, output_dim, vb.pp("fc2"))?,
            gamma: 2.0,
        })
    }
}

impl Module for GateNu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(x)?.relu()?;
        candle_nn::ops::sigmoid(&self.fc2.forward(&hidden)?)?.affine(self.gamma, 0.0)
    }
}
