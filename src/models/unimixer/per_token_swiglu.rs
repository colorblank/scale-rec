//! PerTokenSwiGLU：Token 维度的 SwiGLU 激活与投影。
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

/// Per-Token SwiGLU 模块 (参考论文第 4.3 节)。
///
/// 应用特定于 Token 的 SwiGLU 变换来建模特征的异质性。
/// 公式: pSwiGLU(o_i) = W_down^i * ((W_up^i * o_i + b_up^i) ⊙ Swish(W_gate^i * o_i + b_gate^i)) + b_down^i
pub struct PerTokenSwiGlu {
    w_up_t: Tensor,
    b_up: Tensor,
    w_gate_t: Tensor,
    b_gate: Tensor,
    w_down_t: Tensor,
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

        let w_up_t = w_up.transpose(1, 2)?.contiguous()?;
        let w_gate_t = w_gate.transpose(1, 2)?.contiguous()?;
        let w_down_t = w_down.transpose(1, 2)?.contiguous()?;

        Ok(Self {
            w_up_t,
            b_up,
            w_gate_t,
            b_gate,
            w_down_t,
            b_down,
        })
    }

    /// Swish 激活函数: x * sigmoid(x)
    fn swish(&self, x: &Tensor) -> Result<Tensor> {
        let sig = candle_nn::ops::sigmoid(x)?;
        x.mul(&sig)
    }

    /// 执行 Per-Token 爱因斯坦求和乘法：'btd,thd->bth' (或对应 'bth,thd->btd')
    ///
    /// 通过转置输入并使用批量矩阵乘法绕过，避免了动态复制/广播权重。
    /// - x: shape [B, T, D]
    /// - w_t: shape [T, D, H] (已在初始化时预转置并 contiguous)
    /// 返回: shape [B, T, H]
    fn einsum_btd_thd_bth(x: &Tensor, w_t: &Tensor) -> Result<Tensor> {
        // 1. 转置 X 至 [T, B, D]
        let x_t = x.transpose(0, 1)?.contiguous()?;

        // 2. 批量矩阵乘法 [T, B, D] x [T, D, H] = [T, B, H]
        let out_t = x_t.matmul(w_t)?;

        // 3. 转置回 [B, T, H] 并 contiguous
        out_t.transpose(0, 1)?.contiguous()
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
        let up_proj = Self::einsum_btd_thd_bth(x, &self.w_up_t)?;
        let up = up_proj.broadcast_add(&self.b_up)?;

        // Gate projection (batch_size, T, H)
        let gate_proj = Self::einsum_btd_thd_bth(x, &self.w_gate_t)?;
        let gate = gate_proj.broadcast_add(&self.b_gate)?;

        // Swish activation
        let gate_activated = self.swish(&gate)?;

        // Element-wise mul: up ⊙ Swish(gate) -> (batch_size, T, H)
        let hidden = up.mul(&gate_activated)?;

        // Down projection (batch_size, T, D)
        let down_proj = Self::einsum_btd_thd_bth(&hidden, &self.w_down_t)?;
        let output = down_proj.broadcast_add(&self.b_down)?;

        Ok(output)
    }
}
