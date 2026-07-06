use super::gate::GateNu;
use crate::layers::towers::{Activation, TowerConfig};
use crate::models::OutputKind;
use candle_core::{Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};

pub(super) struct PersonalizedTower {
    pub(super) name: String,
    pub(super) layers: Vec<candle_nn::Linear>,
    pub(super) pp_gates: Vec<GateNu>,
    pub(super) activation: Activation,
    pub(super) output_kind: OutputKind,
}

impl PersonalizedTower {
    pub(super) fn new(
        config: &TowerConfig,
        input_dim: usize,
        prior_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.hidden_dims.len() + 1);
        let mut pp_gates = Vec::with_capacity(config.hidden_dims.len());
        let mut in_dim = input_dim;
        for (i, &h_dim) in config.hidden_dims.iter().enumerate() {
            layers.push(linear(in_dim, h_dim, vb.pp(format!("hidden.{}", i)))?);
            pp_gates.push(GateNu::new(
                vb.pp("pp_gates").pp(i.to_string()),
                prior_dim,
                prior_dim,
                h_dim,
            )?);
            in_dim = h_dim;
        }
        layers.push(linear(
            in_dim,
            config.output_dim,
            vb.pp(format!("output.{}", config.hidden_dims.len())),
        )?);
        Ok(Self {
            name: config.name.clone(),
            layers,
            pp_gates,
            activation: config.activation.clone(),
            output_kind: config.output_kind,
        })
    }

    pub(super) fn forward(&self, shared: &Tensor, prior: &Tensor) -> Result<Tensor> {
        let mut out = shared.clone();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;
            if i < last {
                out = apply_activation(&self.activation, &out)?;
                out = out.broadcast_mul(&self.pp_gates[i].forward(prior)?)?;
            }
        }
        Ok(out)
    }
}

fn apply_activation(activation: &Activation, x: &Tensor) -> Result<Tensor> {
    match activation {
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
