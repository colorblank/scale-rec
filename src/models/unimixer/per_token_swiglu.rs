//! PerTokenSwiGLU：Token 维度的 SwiGLU 激活与投影。
use super::profile;
use candle_core::{Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};
use std::sync::Mutex;

/// Per-Token SwiGLU 模块 (参考论文第 4.3 节)。
///
/// 应用特定于 Token 的 SwiGLU 变换来建模特征的异质性。
/// 公式: pSwiGLU(o_i) = W_down^i * ((W_up^i * o_i + b_up^i) ⊙ Swish(W_gate^i * o_i + b_gate^i)) + b_down^i
struct CachedWeights {
    up_gate_linears: Vec<Linear>,
    down_linears: Vec<Linear>,
}

pub struct PerTokenSwiGlu {
    w_up: Tensor,
    b_up: Tensor,
    w_gate: Tensor,
    b_gate: Tensor,
    w_down: Tensor,
    b_down: Tensor,
    num_tokens: usize,
    token_dim: usize,
    hidden_dim: usize,
    cached_weights: Mutex<Option<CachedWeights>>,
}

impl PerTokenSwiGlu {
    /// 构造 PerTokenSwiGLU 模块
    ///
    /// 参数:
    /// - `num_tokens`: Token 的数量 T
    /// - `token_dim`: Token 维度 D
    /// - `hidden_factor`: 隐藏层维度扩展因子 n (默认一般为 1.0)
    /// - `vb`: 变量构建器
    /// - `down_init_scale`: Down 矩阵初始化缩放系数 (TokenMixer-Large, 默认 1.0)
    pub fn new(
        num_tokens: usize,
        token_dim: usize,
        hidden_factor: f64,
        vb: VarBuilder,
        down_init_scale: f64,
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

        // W_down 权重 (TokenMixer-Large: down_init_scale < 1.0 使初始输出接近零)
        let down_bound = (6.0 / (hidden_dim as f64 + token_dim as f64)).sqrt() * down_init_scale;
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
            num_tokens,
            token_dim,
            hidden_dim,
            cached_weights: Mutex::new(None),
        })
    }

    /// Swish 激活函数: x * sigmoid(x)
    fn swish(&self, x: &Tensor) -> Result<Tensor> {
        let sig = candle_nn::ops::sigmoid(x)?;
        x.mul(&sig)
    }

    fn apply_token_linears(x: &Tensor, linears: &[Linear]) -> Result<Tensor> {
        let (_, num_tokens, _) = x.dims3()?;
        if num_tokens != linears.len() {
            candle_core::bail!(
                "token count mismatch: input has {}, linears has {}",
                num_tokens,
                linears.len()
            );
        }
        let mut outputs = Vec::with_capacity(num_tokens);
        for token_idx in 0..num_tokens {
            let x_token = x.narrow(1, token_idx, 1)?.squeeze(1)?.contiguous()?;
            outputs.push(linears[token_idx].forward(&x_token)?.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }

    fn cached_linears(&self) -> Result<(Vec<Linear>, Vec<Linear>)> {
        let mut guard = self
            .cached_weights
            .lock()
            .expect("PerTokenSwiGlu cache mutex poisoned");
        if let Some(cached) = guard.as_ref() {
            return Ok((cached.up_gate_linears.clone(), cached.down_linears.clone()));
        }
        let up_gate_weight = Tensor::cat(&[self.w_up.clone(), self.w_gate.clone()], 1)?;
        let up_gate_bias = Tensor::cat(&[self.b_up.clone(), self.b_gate.clone()], 2)?.squeeze(0)?;
        let down_bias = self.b_down.squeeze(0)?;
        let num_tokens = up_gate_weight.dim(0)?;
        let mut up_gate_linears = Vec::with_capacity(num_tokens);
        let mut down_linears = Vec::with_capacity(num_tokens);
        for token_idx in 0..num_tokens {
            up_gate_linears.push(Linear::new(
                up_gate_weight.narrow(0, token_idx, 1)?.squeeze(0)?,
                Some(up_gate_bias.narrow(0, token_idx, 1)?.squeeze(0)?),
            ));
            down_linears.push(Linear::new(
                self.w_down.narrow(0, token_idx, 1)?.squeeze(0)?,
                Some(down_bias.narrow(0, token_idx, 1)?.squeeze(0)?),
            ));
        }
        *guard = Some(CachedWeights {
            up_gate_linears: up_gate_linears.clone(),
            down_linears: down_linears.clone(),
        });
        Ok((up_gate_linears, down_linears))
    }

    /// 预构建并缓存每个 token 的线性层权重。
    pub fn warmup(&self) -> Result<()> {
        self.cached_linears().map(|_| ())
    }

    /// 前向传播
    ///
    /// 参数:
    /// - `x`: 形状为 (batch_size, T, D) 的输入张量，其中 T = num_tokens, D = token_dim
    ///
    /// 返回:
    /// - 形状为 (batch_size, T, D) 的输出张量
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let total_timer = profile::start();
        let cache_timer = profile::start();
        let (up_gate_linears, down_linears) = self.cached_linears()?;
        profile::log("pswiglu.cached_linears", cache_timer);

        let up_gate_timer = profile::start();
        let (batch_size, num_tokens, token_dim) = x.dims3()?;
        if num_tokens != self.num_tokens || token_dim != self.token_dim {
            candle_core::bail!(
                "PerTokenSwiGlu input shape mismatch: expected (*, {}, {}), got ({}, {}, {})",
                self.num_tokens,
                self.token_dim,
                batch_size,
                num_tokens,
                token_dim
            );
        }
        let up_gate = Self::apply_token_linears(x, &up_gate_linears)?;
        profile::log("pswiglu.up_gate_linear", up_gate_timer);
        let activation_timer = profile::start();
        let up = up_gate.narrow(2, 0, self.hidden_dim)?;
        let gate = up_gate.narrow(2, self.hidden_dim, self.hidden_dim)?;
        let hidden = up.mul(&self.swish(&gate)?)?;
        profile::log("pswiglu.swish_mul", activation_timer);
        let down_timer = profile::start();
        let output = Self::apply_token_linears(&hidden, &down_linears)?;
        profile::log("pswiglu.down_linear", down_timer);

        profile::log("pswiglu.total", total_timer);
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
        let module = PerTokenSwiGlu::new(2, 3, 1.0, vb, 1.0).unwrap();
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

        assert!(diff <= 1e-5, "max abs diff too large: {diff}");
    }
}
