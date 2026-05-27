//! UniMixingLite：轻量版基组合 + 低秩近似 Token 交互。
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
        let (batch_size, _) = x.dims2()?;
        let n = self.num_blocks;
        let b = self.block_size;
        let l = self.embed_dim;
        let (w_b_star, w_r) = self.cached_mixing(temperature)?;

        // --- Step 1: 将输入划分为块 ---
        let x_blocks = x.reshape((batch_size, n, b))?; // (batch_size, N, B)

        // --- Step 2: 局部交互 ---
        // H = einsum('bnd,nde->bne', x_blocks, W_B_star)
        // 使用批量矩阵乘法实现 [batch_size, N, 1, B] matmul [batch_size, N, B, B]
        let x_blocks_unsqueezed = x_blocks.unsqueeze(2)?.contiguous()?;
        let w_b_star_bcasted = w_b_star
            .unsqueeze(0)?
            .broadcast_as((batch_size, n, b, b))?
            .contiguous()?;
        let h_unsqueezed = x_blocks_unsqueezed.matmul(&w_b_star_bcasted)?;
        let h = h_unsqueezed.squeeze(2)?; // (batch_size, N, B)

        // --- Step 3: 全局交互 ---
        // H_perm = H.permute(0, 2, 1) -> (batch_size, B, N)
        let h_perm = h.transpose(1, 2)?;
        let w_r_t = w_r.transpose(0, 1)?;
        // 批处理矩阵乘法 [batch_size, B, N] matmul [batch_size, N, N]
        let w_r_t_bcasted = w_r_t
            .unsqueeze(0)?
            .broadcast_as((batch_size, n, n))?
            .contiguous()?;
        let h_perm_cont = h_perm.contiguous()?;
        let out_perm = h_perm_cont.matmul(&w_r_t_bcasted)?; // (batch_size, B, N)

        // --- Step 4: 恢复成块的顺序 out_blocks = out_perm.permute(0, 2, 1) -> (batch_size, N, B)
        let out_blocks = out_perm.transpose(1, 2)?;

        // --- Step 5: 恢复为原始输出维度 ---
        let out = out_blocks.reshape((batch_size, l))?;

        Ok(out)
    }
}
