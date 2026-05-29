//! PerTokenSwiGLU：Token 维度的 SwiGLU 激活与投影。
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

/// Per-Token SwiGLU 模块 (参考论文第 4.3 节)。
///
/// 应用特定于 Token 的 SwiGLU 变换来建模特征的异质性。
/// 公式: pSwiGLU(o_i) = W_down^i * ((W_up^i * o_i + b_up^i) ⊙ Swish(W_gate^i * o_i + b_gate^i)) + b_down^i
pub struct PerTokenSwiGlu {
    w_up: Tensor,
    b_up: Tensor,
    w_gate: Tensor,
    b_gate: Tensor,
    w_down: Tensor,
    b_down: Tensor,
}

impl PerTokenSwiGlu {
    /// 构造 PerTokenSwiGLU 模块
    ///
    /// 参数:
    /// - `num_tokens`: Token 的数量 T
    /// - `token_dim`: Token 维度 D
    /// - `hidden_factor`: 隐藏层维度扩展因子 n (默认一般为 1.0)
    /// - `vb`: 变量构建器
    pub fn new(
        num_tokens: usize,
        token_dim: usize,
        hidden_factor: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_dim = (token_dim as f64 * hidden_factor) as usize;
        if hidden_dim == 0 {
            candle_core::bail!("hidden_factor produces zero hidden dimension");
        }

        // 使用 Xavier uniform 初始化 (对应 PyTorch 的 xavier_uniform_)
        // 计算 W_up 权重的边界界限
        let up_bound = (6.0 / (token_dim as f64 + hidden_dim as f64)).sqrt();
        let w_up = vb.get_with_hints(
            (num_tokens, hidden_dim, token_dim),
            "w_up",
            Init::Uniform {
                lo: -up_bound,
                up: up_bound,
            },
        )?;
        let b_up = vb.get_with_hints((1, num_tokens, hidden_dim), "b_up", Init::Const(0.0))?;

        // W_gate 权重
        let gate_bound = (6.0 / (token_dim as f64 + hidden_dim as f64)).sqrt();
        let w_gate = vb.get_with_hints(
            (num_tokens, hidden_dim, token_dim),
            "w_gate",
            Init::Uniform {
                lo: -gate_bound,
                up: gate_bound,
            },
        )?;
        let b_gate = vb.get_with_hints((1, num_tokens, hidden_dim), "b_gate", Init::Const(0.0))?;

        // W_down 权重
        let down_bound = (6.0 / (hidden_dim as f64 + token_dim as f64)).sqrt();
        let w_down = vb.get_with_hints(
            (num_tokens, token_dim, hidden_dim),
            "w_down",
            Init::Uniform {
                lo: -down_bound,
                up: down_bound,
            },
        )?;
        let b_down = vb.get_with_hints((1, num_tokens, token_dim), "b_down", Init::Const(0.0))?;

        Ok(Self {
            w_up,
            b_up,
            w_gate,
            b_gate,
            w_down,
            b_down,
        })
    }

    /// Swish 激活函数: x * sigmoid(x)
    fn swish(&self, x: &Tensor) -> Result<Tensor> {
        let sig = candle_nn::ops::sigmoid(x)?;
        x.mul(&sig)
    }

    /// 执行 Per-Token 乘法：`[B, T, D] x [T, H, D] -> [B, T, H]`
    ///
    /// 这里保持数学定义不变，只把实现改成按 token 维度的 batch matmul，避免
    /// 将权重展开成 `[B, T, D, H]` 带来的额外拷贝和内存带宽开销。
    fn einsum_btd_thd_bth(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        // 令 batch 维度落在最前面，和 token 维一起做 batch matmul。
        // x_t: [T, B, D], w_t: [T, D, H] => out_t: [T, B, H]
        let x_t = x.transpose(0, 1)?.contiguous()?;
        let w_t = w.transpose(1, 2)?.contiguous()?;
        let out_t = x_t.matmul(&w_t)?;

        // 还原成 [B, T, H]
        out_t.transpose(0, 1)
    }

    /// 前向传播
    ///
    /// 参数:
    /// - `x`: 形状为 (batch_size, T, D) 的输入张量，其中 T = num_tokens, D = token_dim
    ///
    /// 返回:
    /// - 形状为 (batch_size, T, D) 的输出张量
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Up projection (batch_size, T, H)
        let up_proj = Self::einsum_btd_thd_bth(x, &self.w_up)?;
        let up = up_proj.broadcast_add(&self.b_up)?;

        // Gate projection (batch_size, T, H)
        let gate_proj = Self::einsum_btd_thd_bth(x, &self.w_gate)?;
        let gate = gate_proj.broadcast_add(&self.b_gate)?;

        // Swish activation
        let gate_activated = self.swish(&gate)?;

        // Element-wise mul: up ⊙ Swish(gate) -> (batch_size, T, H)
        let hidden = up.mul(&gate_activated)?;

        // Down projection (batch_size, T, D)
        let down_proj = Self::einsum_btd_thd_bth(&hidden, &self.w_down)?;
        let output = down_proj.broadcast_add(&self.b_down)?;

        Ok(output)
    }
}

#[cfg(test)]
impl PerTokenSwiGlu {
    /// 参考实现：保留原先的广播式 batch matmul，专用于测试新旧结果一致。
    fn einsum_btd_thd_bth_reference(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (batch_size, t, d) = x.dims3()?;
        let (_, h, _) = w.dims3()?;

        let w_t = w.transpose(1, 2)?;
        let x_unsqueezed = x.unsqueeze(2)?.contiguous()?;
        let w_bcasted = w_t
            .unsqueeze(0)?
            .broadcast_as((batch_size, t, d, h))?
            .contiguous()?;
        let out_unsqueezed = x_unsqueezed.matmul(&w_bcasted)?;
        out_unsqueezed.squeeze(2)
    }

    fn forward_reference(&self, x: &Tensor) -> Result<Tensor> {
        let up_proj = Self::einsum_btd_thd_bth_reference(x, &self.w_up)?;
        let up = up_proj.broadcast_add(&self.b_up)?;

        let gate_proj = Self::einsum_btd_thd_bth_reference(x, &self.w_gate)?;
        let gate = gate_proj.broadcast_add(&self.b_gate)?;
        let gate_activated = self.swish(&gate)?;

        let hidden = up.mul(&gate_activated)?;
        let down_proj = Self::einsum_btd_thd_bth_reference(&hidden, &self.w_down)?;
        down_proj.broadcast_add(&self.b_down)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn make_module() -> PerTokenSwiGlu {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let module = PerTokenSwiGlu::new(2, 3, 1.0, vb).unwrap();
        // Keep varmap alive for the module lifetime in tests.
        let _ = Box::leak(Box::new(varmap));
        module
    }

    #[test]
    fn optimized_forward_matches_reference() {
        let module = make_module();
        let device = Device::Cpu;

        let x = Tensor::from_slice(
            &[
                0.5f32, -1.0, 2.0, //
                1.5, 0.0, -0.5, //
                -2.0, 3.0, 0.25, //
                4.0, -3.5, 1.25,
            ],
            (2, 2, 3),
            &device,
        )
        .unwrap();

        let fast = module.forward(&x).unwrap();
        let reference = module.forward_reference(&x).unwrap();
        let diff = fast
            .sub(&reference)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(diff <= 1e-6, "max abs diff too large: {diff}");
    }
}
