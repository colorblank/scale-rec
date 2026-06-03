//! UniMixingLite：轻量版基组合 + 低秩近似 Token 交互。
use super::profile;
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};
use std::sync::Mutex;

/// 轻量级 UniMixing 模块 (UniMixing-Lite, 参考论文第 4.3 节公式 18)。
///
/// 对于局部混合使用基组合 (basis composition)，
/// 对于全局混合使用低秩近似 (low-rank approximation)。
pub struct UniMixingLite {
    pub embed_dim: usize,
    pub block_size: usize,
    pub num_basis: usize,
    pub rank: usize,
    pub num_blocks: usize,

    /// 全局混合参数 A_G (N, r)
    a_g: Tensor,
    /// 全局混合参数 B_G (r, N)
    b_g: Tensor,
    /// 局部混合基矩阵 Z_l (b, B, B)
    z: Tensor,
    /// 局部混合块特定权重 omega (N, b)
    omega: Tensor,
    cached_mixing: Mutex<Option<CachedMixing>>,
}

struct CachedMixing {
    temperature_bits: u64,
    w_b_star: Tensor,
    w_r: Tensor,
}

impl UniMixingLite {
    /// 构造 UniMixingLite 模块
    ///
    /// 参数:
    /// - `embed_dim`: 输入特征维度 L
    /// - `block_size`: 块大小 B。L 必须能被 B 整除。
    /// - `num_basis`: 局部混合基矩阵的数量 b (默认 4)
    /// - `rank`: 全局混合低秩近似的秩 r (默认 16)
    /// - `vb`: 变量构建器
    pub fn new(
        embed_dim: usize,
        block_size: usize,
        num_basis: usize,
        rank: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if block_size == 0 {
            candle_core::bail!("block_size must be > 0");
        }
        if embed_dim % block_size != 0 {
            candle_core::bail!(
                "embed_dim ({}) 必须能被 block_size ({}) 整除",
                embed_dim,
                block_size
            );
        }
        if num_basis == 0 {
            candle_core::bail!("num_basis must be > 0");
        }
        if rank == 0 {
            candle_core::bail!("rank must be > 0");
        }
        let num_blocks = embed_dim / block_size;

        // 全局混合低秩矩阵
        let a_g = vb.get_with_hints(
            (num_blocks, rank),
            "a_g",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let b_g = vb.get_with_hints(
            (rank, num_blocks),
            "b_g",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        // 局部混合基组合
        let z = vb.get_with_hints(
            (num_basis, block_size, block_size),
            "z",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let omega = vb.get_with_hints(
            (num_blocks, num_basis),
            "omega",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        Ok(Self {
            embed_dim,
            block_size,
            num_basis,
            rank,
            num_blocks,
            a_g,
            b_g,
            z,
            omega,
            cached_mixing: Mutex::new(None),
        })
    }

    /// 应用 Sinkhorn-Knopp 迭代生成双随机矩阵
    fn sinkhorn_knopp(&self, matrix: &Tensor, n_iters: usize) -> Result<Tensor> {
        let max = matrix
            .max_keepdim(matrix.rank() - 1)?
            .max_keepdim(matrix.rank() - 2)?;
        let mut mat = matrix.broadcast_sub(&max)?.exp()?;
        let eps = 1e-8f64;
        let rank = mat.rank();

        for _ in 0..n_iters {
            // 行归一化
            let sum_row = (mat.sum_keepdim(rank - 1)? + eps)?;
            mat = mat.broadcast_div(&sum_row)?;

            // 列归一化
            let sum_col = (mat.sum_keepdim(rank - 2)? + eps)?;
            mat = mat.broadcast_div(&sum_col)?;
        }
        Ok(mat)
    }

    fn build_cached_mixing(&self, temperature: f64) -> Result<CachedMixing> {
        // w_b_logits[i, j, k] = sum_m omega[i, m] * z[m, j, k]
        // 这等价于把 z 展平后做一次矩阵乘法，避免广播到 4D 张量。
        let z_flat = self
            .z
            .reshape((self.num_basis, self.block_size * self.block_size))?;
        let w_b_logits = self.omega.matmul(&z_flat)?.reshape((
            self.num_blocks,
            self.block_size,
            self.block_size,
        ))?;
        let w_b_t = w_b_logits.transpose(1, 2)?;
        let w_b_sym = ((w_b_logits + w_b_t)? * 0.5)?;
        let w_b_div = (&w_b_sym * (1.0 / temperature))?;
        let w_b_star = self.sinkhorn_knopp(&w_b_div, 3)?;

        let w_g_logits = self.a_g.matmul(&self.b_g)?;
        let w_g_t = w_g_logits.transpose(0, 1)?;
        let w_g_sym = ((w_g_logits + w_g_t)? * 0.5)?;
        let w_g_div = (&w_g_sym * (1.0 / temperature))?;
        let w_r = self.sinkhorn_knopp(&w_g_div, 3)?;

        Ok(CachedMixing {
            temperature_bits: temperature.to_bits(),
            w_b_star,
            w_r,
        })
    }

    fn cached_mixing(&self, temperature: f64) -> Result<(Tensor, Tensor)> {
        let temperature_bits = temperature.to_bits();
        if let Some((w_b_star, w_r)) = self
            .cached_mixing
            .lock()
            .expect("UniMixingLite cache mutex poisoned")
            .as_ref()
            .and_then(|cached| {
                (cached.temperature_bits == temperature_bits)
                    .then(|| (cached.w_b_star.clone(), cached.w_r.clone()))
            })
        {
            return Ok((w_b_star, w_r));
        }

        let cached = self.build_cached_mixing(temperature)?;
        let w_b_star = cached.w_b_star.clone();
        let w_r = cached.w_r.clone();
        let mut guard = self
            .cached_mixing
            .lock()
            .expect("UniMixingLite cache mutex poisoned");
        if guard
            .as_ref()
            .map(|cached| cached.temperature_bits != temperature_bits)
            .unwrap_or(true)
        {
            *guard = Some(cached);
        }
        Ok((w_b_star, w_r))
    }

    pub fn warmup(&self, temperature: f64) -> Result<()> {
        self.cached_mixing(temperature).map(|_| ())
    }

    fn local_mix_2d_loop(&self, x: &Tensor, w_b_star: &Tensor) -> Result<Tensor> {
        let (batch_size, _) = x.dims2()?;
        let x_blocks = x.reshape((batch_size, self.num_blocks, self.block_size))?;
        let mut outputs = Vec::with_capacity(self.num_blocks);
        for block_idx in 0..self.num_blocks {
            let x_block = x_blocks.narrow(1, block_idx, 1)?.squeeze(1)?.contiguous()?;
            let w_block = w_b_star.narrow(0, block_idx, 1)?.squeeze(0)?;
            outputs.push(x_block.matmul(&w_block)?.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }

    /// 前向传播
    ///
    /// 参数:
    /// - `x`: 输入张量，形状为 (batch_size, L)
    /// - `temperature`: 退火温度系数
    ///
    /// 返回:
    /// - 输出张量，形状为 (batch_size, L)
    pub fn forward(&self, x: &Tensor, temperature: f64) -> Result<Tensor> {
        if temperature <= 0.0 {
            candle_core::bail!("temperature must be > 0");
        }
        let total_timer = profile::start();
        let (batch_size, _) = x.dims2()?;
        let n = self.num_blocks;
        let b = self.block_size;
        let l = self.embed_dim;
        let cache_timer = profile::start();
        let (w_b_star, w_r) = self.cached_mixing(temperature)?;
        profile::log("unimixing_lite.cached_mixing", cache_timer);

        // --- Step 1: 局部交互 ---
        // [batch_size, N, B] × [N, B, B] → [batch_size, N, B]。
        // 避免 Candle native CPU 后端的 3D batched matmul 慢路径。
        let local_timer = profile::start();
        let h = self.local_mix_2d_loop(x, &w_b_star)?;
        profile::log("unimixing_lite.local_mix", local_timer);

        // --- Step 2: 全局交互 ---
        let global_timer = profile::start();
        let h_flat = h
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, batch_size * b))?;
        let out_flat = w_r.matmul(&h_flat)?;
        profile::log("unimixing_lite.global_mix", global_timer);

        // --- Step 3: 恢复输出维度 ---
        let reshape_timer = profile::start();
        let out = out_flat
            .reshape((n, batch_size, b))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((batch_size, l))?;
        profile::log("unimixing_lite.output_reshape", reshape_timer);

        profile::log("unimixing_lite.total", total_timer);
        Ok(out)
    }
}

#[cfg(test)]
impl UniMixingLite {
    fn build_cached_mixing_reference(&self, temperature: f64) -> Result<(Tensor, Tensor)> {
        let omega_exp = self.omega.unsqueeze(2)?.unsqueeze(3)?;
        let z_exp = self.z.unsqueeze(0)?;
        let w_b_logits = omega_exp.broadcast_mul(&z_exp)?.sum(1)?;
        let w_b_t = w_b_logits.transpose(1, 2)?;
        let w_b_sym = ((w_b_logits + w_b_t)? * 0.5)?;
        let w_b_div = (&w_b_sym * (1.0 / temperature))?;
        let w_b_star = self.sinkhorn_knopp(&w_b_div, 3)?;

        let w_g_logits = self.a_g.matmul(&self.b_g)?;
        let w_g_t = w_g_logits.transpose(0, 1)?;
        let w_g_sym = ((w_g_logits + w_g_t)? * 0.5)?;
        let w_g_div = (&w_g_sym * (1.0 / temperature))?;
        let w_r = self.sinkhorn_knopp(&w_g_div, 3)?;

        Ok((w_b_star, w_r))
    }

    fn forward_reference(&self, x: &Tensor, temperature: f64) -> Result<Tensor> {
        if temperature <= 0.0 {
            candle_core::bail!("temperature must be > 0");
        }
        let (batch_size, _) = x.dims2()?;
        let n = self.num_blocks;
        let b = self.block_size;
        let l = self.embed_dim;
        let (w_b_star, w_r) = self.build_cached_mixing_reference(temperature)?;

        let x_blocks = x.reshape((batch_size, n, b))?;
        let x_blocks_t = x_blocks.transpose(0, 1)?.contiguous()?;
        let h_t = x_blocks_t.matmul(&w_b_star)?;
        let h = h_t.transpose(0, 1)?;

        let h_trans = h.transpose(0, 1)?.contiguous()?;
        let h_reshaped = h_trans.reshape((n, batch_size * b))?;
        let out_reshaped = w_r.matmul(&h_reshaped)?;
        let out_trans = out_reshaped.reshape((n, batch_size, b))?;
        let out_blocks = out_trans.transpose(0, 1)?;

        out_blocks.contiguous()?.reshape((batch_size, l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn make_module() -> UniMixingLite {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let module = UniMixingLite::new(8, 4, 2, 3, vb).unwrap();
        let _ = Box::leak(Box::new(varmap));
        module
    }

    #[test]
    fn optimized_forward_matches_reference() {
        let module = make_module();
        let device = Device::Cpu;
        let x = Tensor::from_slice(
            &[
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, //
                1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, //
            ],
            (2, 8),
            &device,
        )
        .unwrap();

        let fast = module.forward(&x, 1.0).unwrap();
        let reference = module.forward_reference(&x, 1.0).unwrap();
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
