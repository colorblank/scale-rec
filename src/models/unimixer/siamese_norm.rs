use candle_core::{Result, Tensor};
use candle_nn::{rms_norm, Module, RmsNorm, VarBuilder};

/// SiameseNorm 输出类型的枚举，处理两种不同的返回状态
pub enum SiameseNormOutput {
    /// 包含更新后的主流和辅助流 (X_bar_new, Y_bar_new)
    Streams(Tensor, Tensor),
    /// 包含最终融合后的单一张量
    Fused(Tensor),
}

/// SiameseNorm 模块 (参考论文 4.4 节, 公式 19-20)。
///
/// 通过每层引入两个耦合的流，解决了 Pre-Norm 和 Post-Norm 之间的张力。
pub struct SiameseNorm {
    rmsnorm: RmsNorm,
}

impl SiameseNorm {
    /// 构造 SiameseNorm 模块
    ///
    /// 参数:
    /// - `normalized_shape`: 归一化的特征维度大小
    /// - `eps`: 归一化中的数值稳定性常数
    /// - `vb`: 变量构建器
    pub fn new(normalized_shape: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let rmsnorm = rms_norm(normalized_shape, eps, vb.pp("rmsnorm"))?;
        Ok(Self { rmsnorm })
    }

    pub fn forward_rmsnorm(&self, x: &Tensor) -> Result<Tensor> {
        self.rmsnorm.forward(x)
    }

    pub fn forward(
        &self,
        x_bar: &Tensor,
        y_bar: &Tensor,
        output: Option<&Tensor>,
    ) -> Result<SiameseNormOutput> {
        if let Some(out) = output {
            // 更新流 Update streams (Eq. 19)
            // X_bar_new = RMSNorm(X_bar + output)
            let x_bar_added = x_bar.broadcast_add(out)?;
            let x_bar_new = self.rmsnorm.forward(&x_bar_added)?;

            // Y_bar_new = Y_bar + output
            let y_bar_new = y_bar.broadcast_add(out)?;

            Ok(SiameseNormOutput::Streams(x_bar_new, y_bar_new))
        } else {
            // 最终融合 Final fusion (Eq. 20)
            // output_fused = X_bar + RMSNorm(Y_bar)
            let y_bar_norm = self.rmsnorm.forward(y_bar)?;
            let output_fused = x_bar.broadcast_add(&y_bar_norm)?;

            Ok(SiameseNormOutput::Fused(output_fused))
        }
    }
}
