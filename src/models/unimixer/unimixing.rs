//! UniMixing：标准版双随机矩阵 Token 交互。
use candle_core::{Result, Tensor};
use candle_nn::{Init, VarBuilder};

/// 标准 `UniMixing` 模块 (参考论文第 4.3 节)。
///
/// 实现了广义的参数化 Token 混合操作。
/// 核心思想是将混合过程解耦为局部的块内交互 (Local Interaction) 和全局的块间交互 (Global Interaction)，
/// 以大幅降低注意力机制的时间和空间复杂度。
pub struct UniMixing {
    pub embed_dim: usize,
    pub block_size: usize,
    pub num_blocks: usize,
    /// 全局交互权重 `W_G` (形状: `[num_blocks, num_blocks]`)
    global_weights_logits: Tensor,
    /// 局部交互权重 `W_B` (形状: `[num_blocks, block_size, block_size]`)
    local_weights_logits: Tensor,
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
        let (batch_size, _) = x.dims2()?;
        let n = self.num_blocks;
        let b = self.block_size;

        // --- 1. 分割输入特征矩阵 ---
        // X: [batch_size, L] -> [batch_size, N, B]
        let x_blocks = x.reshape((batch_size, n, b))?;

        // --- 2. 局部交互 (Local Interaction) ---
        // 生成双随机矩阵并应用对称约束: (W + W^T) / 2
        let w_b_div = (&self.local_weights_logits * (1.0 / temperature))?;
        let w_b_sink = self.sinkhorn_knopp(&w_b_div, 3)?;
        let w_b_t = w_b_sink.transpose(1, 2)?;
        let w_b_proc = ((w_b_sink + w_b_t)? * 0.5)?;

        // 使用 Batch Matmul 计算局部混合: H = x_blocks * W_B
        let x_blocks_unsqueezed = x_blocks.unsqueeze(2)?.contiguous()?;
        let w_b_proc_bcasted = w_b_proc
            .unsqueeze(0)?
            .broadcast_as((batch_size, n, b, b))?
            .contiguous()?;

        let h_unsqueezed = x_blocks_unsqueezed.matmul(&w_b_proc_bcasted)?;
        let h = h_unsqueezed.squeeze(2)?;

        // --- 3. 全局交互 (Global Interaction) ---
        // 生成全局双随机矩阵并应用对称约束
        let w_g_div = (&self.global_weights_logits * (1.0 / temperature))?;
        let w_g_sink = self.sinkhorn_knopp(&w_g_div, 3)?;
        let w_g_t = w_g_sink.transpose(0, 1)?;
        let w_g_proc = ((w_g_sink + w_g_t)? * 0.5)?;

        // 使用 Batch Matmul 计算跨块全局混合: out_blocks = W_G * H
        // W_G: [batch_size, N, N], H: [batch_size, N, B] -> [batch_size, N, B]
        let w_g_proc_bcasted = w_g_proc
            .unsqueeze(0)?
            .broadcast_as((batch_size, n, n))?
            .contiguous()?;
        let h_contiguous = h.contiguous()?;
        let out_blocks = w_g_proc_bcasted.matmul(&h_contiguous)?;

        // --- 4. 恢复输出维度 ---
        // [batch_size, N, B] -> [batch_size, L]
        let out = out_blocks.reshape((batch_size, self.embed_dim))?;

        Ok(out)
    }
}
