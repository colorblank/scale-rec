//! Gated Deep & Cross Network layers.
use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

/// Gated cross network over dense feature vectors.
pub struct GatedCrossNetwork {
    cross: Vec<Linear>,
    gate: Vec<Linear>,
    bias: Vec<Tensor>,
}

impl GatedCrossNetwork {
    pub fn new(vb: VarBuilder, input_dim: usize, num_layers: usize) -> Result<Self> {
        if input_dim == 0 {
            candle_core::bail!("input_dim must be positive");
        }
        if num_layers == 0 {
            candle_core::bail!("num_layers must be positive");
        }
        let mut cross = Vec::with_capacity(num_layers);
        let mut gate = Vec::with_capacity(num_layers);
        let mut bias = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            cross.push(linear_no_bias(
                input_dim,
                input_dim,
                vb.pp(format!("cross.{}", i)),
            )?);
            gate.push(linear_no_bias(
                input_dim,
                input_dim,
                vb.pp(format!("gate.{}", i)),
            )?);
            bias.push(vb.get(input_dim, &format!("bias.{}", i))?);
        }
        Ok(Self { cross, gate, bias })
    }
}

impl Module for GatedCrossNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x0 = x;
        let mut xi = x.clone();
        for i in 0..self.cross.len() {
            let weight = candle_nn::ops::sigmoid(&self.gate[i].forward(&xi)?)?;
            let crossed = self.cross[i].forward(&xi)?.broadcast_add(&self.bias[i])?;
            xi = x0.mul(&crossed)?.mul(&weight)?.broadcast_add(&xi)?;
        }
        Ok(xi)
    }
}
