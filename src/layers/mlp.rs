//! Mlp：通用多层感知机，层间带激活，末层无激活。
use super::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

pub struct Mlp {
    layers: Vec<Linear>,
    activation: Activation,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl Mlp {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        activation: Activation,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(hidden_dims.len() + 1);
        let mut in_dim = input_dim;
        for (i, &h_dim) in hidden_dims.iter().enumerate() {
            layers.push(linear(in_dim, h_dim, vb.pp(format!("hidden.{}", i)))?);
            in_dim = h_dim;
        }
        layers.push(linear(in_dim, output_dim, vb.pp("output"))?);
        Ok(Self {
            layers,
            activation,
            input_dim,
            output_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;
            if i < n - 1 {
                out = self.apply_activation(&out)?;
            }
        }
        Ok(out)
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::Relu => x.relu(),
            Activation::Sigmoid => candle_nn::ops::sigmoid(x),
            Activation::Swish => {
                let sig = candle_nn::ops::sigmoid(x)?;
                x.mul(&sig)
            }
            Activation::Gelu => x.gelu(),
            Activation::None_ => Ok(x.clone()),
        }
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
