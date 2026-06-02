//! Gated Deep & Cross Network layers.
use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

/// Gated cross network over dense feature vectors.
pub struct GatedCrossNetwork {
    layers: Vec<GatedCrossLayer>,
}

struct GatedCrossLayer {
    cross_gate_weight: Tensor,
    bias: Tensor,
    input_dim: usize,
}

impl GatedCrossNetwork {
    pub fn new(vb: VarBuilder, input_dim: usize, num_layers: usize) -> Result<Self> {
        if input_dim == 0 {
            candle_core::bail!("input_dim must be positive");
        }
        if num_layers == 0 {
            candle_core::bail!("num_layers must be positive");
        }
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let cross = linear_no_bias(input_dim, input_dim, vb.pp(format!("cross.{}", i)))?;
            let gate = linear_no_bias(input_dim, input_dim, vb.pp(format!("gate.{}", i)))?;
            let bias = vb.get(input_dim, &format!("bias.{}", i))?;
            layers.push(GatedCrossLayer::new(cross, gate, bias, input_dim)?);
        }
        Ok(Self { layers })
    }
}

impl GatedCrossLayer {
    fn new(cross: Linear, gate: Linear, bias: Tensor, input_dim: usize) -> Result<Self> {
        let cross_gate_weight = Tensor::cat(&[cross.weight(), gate.weight()], 0)?;
        Ok(Self {
            cross_gate_weight,
            bias,
            input_dim,
        })
    }

    fn forward(&self, x0: &Tensor, xi: &Tensor) -> Result<Tensor> {
        let combined = xi.matmul(&self.cross_gate_weight.t()?)?;
        let last_dim = combined.rank() - 1;
        let crossed = combined
            .narrow(last_dim, 0, self.input_dim)?
            .broadcast_add(&self.bias)?;
        let gate = combined.narrow(last_dim, self.input_dim, self.input_dim)?;
        let weight = candle_nn::ops::sigmoid(&gate)?;
        x0.mul(&crossed)?.mul(&weight)?.broadcast_add(xi)
    }
}

impl Module for GatedCrossNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x0 = x;
        let mut xi = x.clone();
        for layer in &self.layers {
            xi = layer.forward(x0, &xi)?;
        }
        Ok(xi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn fused_cross_gate_matches_separate_linear_outputs() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let cross = linear_no_bias(3, 3, vb.pp("cross"))?;
        let gate = linear_no_bias(3, 3, vb.pp("gate"))?;
        let bias = Tensor::new(&[0.1f32, -0.2, 0.3], &device)?;
        let layer = GatedCrossLayer::new(cross.clone(), gate.clone(), bias, 3)?;
        let x = Tensor::new(&[[0.2f32, 0.4, 0.6], [0.8, 1.0, 1.2]], &device)?;

        let combined = x.matmul(&layer.cross_gate_weight.t()?)?;
        let fused_cross = combined.narrow(1, 0, 3)?;
        let fused_gate = combined.narrow(1, 3, 3)?;

        let cross_diff = fused_cross.sub(&cross.forward(&x)?)?.abs()?.max_all()?;
        let gate_diff = fused_gate.sub(&gate.forward(&x)?)?.abs()?.max_all()?;

        assert!(cross_diff.to_scalar::<f32>()? <= 1e-6);
        assert!(gate_diff.to_scalar::<f32>()? <= 1e-6);
        Ok(())
    }
}
