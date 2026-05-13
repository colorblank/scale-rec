//! FM 二阶交互：0.5 * Σ[(Σv_i)² - Σv_i²]。
use candle_core::{Result, Tensor};

pub fn fm_interaction(stacked: &Tensor) -> Result<Tensor> {
    let sum_square = stacked.sqr()?.sum(1)?;
    let square_sum = stacked.sum(1)?.sqr()?;
    Ok((sum_square.broadcast_sub(&square_sum)?.sum_keepdim(1)? * 0.5)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_fm_interaction_shape() {
        let t = Tensor::randn(0f32, 1f32, (2, 3, 4), &Device::Cpu).unwrap();
        let result = fm_interaction(&t).unwrap();
        assert_eq!(result.dims(), &[2, 1]);
    }
}
