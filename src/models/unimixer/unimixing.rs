//! UniMixing：标准版双随机矩阵 Token 交互。
use super::profile;
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};
use std::sync::Mutex;

/// 标准 `UniMixing` 模块 (参考论文第 4.3 节)。
///
/// 实现了广义的参数化 Token 混合操作。
/// 核心思想是将混合过程解耦为局部的块内交互 (Local Interaction) 和全局的块间交互 (Global Interaction)，
/// 以大幅降低注意力机制的时间和空间复杂度。
pub struct UniMixing {
    /// token 序列展平后的总维度。
    pub embed_dim: usize,
    /// 块内 token 数量。
    pub block_size: usize,
    /// 块数量。
    pub num_blocks: usize,
    /// 全局交互权重 `W_G` (形状: `[num_blocks, num_blocks]`)
    global_weights_logits: Tensor,
    /// 局部交互权重 `W_B` (形状: `[num_blocks, block_size, block_size]`)
    local_weights_logits: Tensor,
    cached_mixing: Mutex<Option<CachedMixing>>,
}

struct CachedMixing {
    temperature_bits: u64,
    w_b_star: Tensor,
    w_r: Tensor,
}

impl UniMixing {
    /// 构造 `UniMixing` 模块。
    ///
    /// # 参数
    /// - `embed_dim`: 输入特征的展平总维度 (L)。
    /// - `block_size`: 块的大小 (B)。注意：`embed_dim` 必须能被 `block_size` 整除。
    /// - `vb`: 变量构建器，用于张量权重的注册和初始化。
    pub fn new(embed_dim: usize, block_size: usize, vb: VarBuilder) -> Result<Self> {
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
        let num_blocks = embed_dim / block_size;

        // 全局交互权重 W_G，形状: (num_blocks, num_blocks)
        let global_weights_logits = vb.get_with_hints(
            (num_blocks, num_blocks),
            "global_weights_logits",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        // 局部交互权重 W_B，形状: (num_blocks, block_size, block_size)
        let local_weights_logits = vb.get_with_hints(
            (num_blocks, block_size, block_size),
            "local_weights_logits",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;

        Ok(Self {
            embed_dim,
            block_size,
            num_blocks,
            global_weights_logits,
            local_weights_logits,
            cached_mixing: Mutex::new(None),
        })
    }

    /// 应用 `Sinkhorn-Knopp` 迭代算法生成双随机矩阵 (doubly stochastic matrix)。
    ///
    /// 通过交替对行和列进行归一化，使得矩阵的每一行和每一列的总和均近似为 1，
    /// 以提供一种具备结构归纳偏置的稀疏注意力替代方案。
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
        let w_b_div = (&self.local_weights_logits * (1.0 / temperature))?;
        let w_b_sink = self.sinkhorn_knopp(&w_b_div, 3)?;
        let w_b_t = w_b_sink.transpose(1, 2)?;
        let w_b_star = ((w_b_sink + w_b_t)? * 0.5)?;

        let w_g_div = (&self.global_weights_logits * (1.0 / temperature))?;
        let w_g_sink = self.sinkhorn_knopp(&w_g_div, 3)?;
        let w_g_t = w_g_sink.transpose(0, 1)?;
        let w_r = ((w_g_sink + w_g_t)? * 0.5)?;

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
            .expect("UniMixing cache mutex poisoned")
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
            .expect("UniMixing cache mutex poisoned");
        if guard
            .as_ref()
            .map(|cached| cached.temperature_bits != temperature_bits)
            .unwrap_or(true)
        {
            *guard = Some(cached);
        }
        Ok((w_b_star, w_r))
    }

    /// 按给定温度预计算并缓存 mixing 矩阵。
    pub fn warmup(&self, temperature: f64) -> Result<()> {
        self.cached_mixing(temperature).map(|_| ())
    }

    fn local_mix_2d_loop(&self, x: &Tensor, w_b_proc: &Tensor) -> Result<Tensor> {
        let (batch_size, _) = x.dims2()?;
        let x_blocks = x.reshape((batch_size, self.num_blocks, self.block_size))?;
        let mut outputs = Vec::with_capacity(self.num_blocks);
        for block_idx in 0..self.num_blocks {
            let x_block = x_blocks.narrow(1, block_idx, 1)?.squeeze(1)?.contiguous()?;
            let w_block = w_b_proc.narrow(0, block_idx, 1)?.squeeze(0)?;
            outputs.push(x_block.matmul(&w_block)?.unsqueeze(1)?);
        }
        Tensor::cat(&outputs, 1)
    }

    /// 执行前向传播。
    ///
    /// # 参数
    /// - `x`: 输入特征张量，形状为 `[batch_size, embed_dim]`。
    /// - `temperature`: 退火温度系数 (τ)。较低的温度会使生成的混合矩阵更趋近于稀疏/硬对齐。
    ///
    /// # 返回值
    /// 返回经过局部和全局交互混合后的张量，形状为 `[batch_size, embed_dim]`。
    pub fn forward(&self, x: &Tensor, temperature: f64) -> Result<Tensor> {
        if temperature <= 0.0 {
            candle_core::bail!("temperature must be > 0");
        }
        let total_timer = profile::start();
        let (batch_size, _) = x.dims2()?;
        let n = self.num_blocks;
        let b = self.block_size;
        let cache_timer = profile::start();
        let (w_b_proc, w_g_proc) = self.cached_mixing(temperature)?;
        profile::log("unimixing.cached_mixing", cache_timer);

        // --- 1. 局部交互 ---
        // [batch_size, N, B] × [N, B, B] → [batch_size, N, B]。
        // 避免 Candle native CPU 后端的 3D batched matmul 慢路径。
        let local_timer = profile::start();
        let h = self.local_mix_2d_loop(x, &w_b_proc)?;
        profile::log("unimixing.local_mix", local_timer);

        // --- 2. 全局交互 ---
        let global_timer = profile::start();
        let h_flat = h
            .transpose(0, 1)?
            .contiguous()?
            .reshape((n, batch_size * b))?;
        let out_flat = w_g_proc.matmul(&h_flat)?;
        profile::log("unimixing.global_mix", global_timer);

        // --- 3. 恢复输出维度 ---
        // [N, batch_size, B] → [batch_size, N, B] → [batch_size, L]
        let reshape_timer = profile::start();
        let out = out_flat
            .reshape((n, batch_size, b))?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((batch_size, self.embed_dim))?;
        profile::log("unimixing.output_reshape", reshape_timer);

        profile::log("unimixing.total", total_timer);
        Ok(out)
    }
}
