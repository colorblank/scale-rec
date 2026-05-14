//! FM 二阶交互：0.5 * Σ[(Σv_i)² - Σv_i²]。
use candle_core::{Result, Tensor};

pub fn fm_interaction(stacked: &Tensor) -> Result<Tensor> {
    let sum_square = stacked.sqr()?.sum(1)?;
    let square_sum = stacked.sum(1)?.sqr()?;
    Ok((square_sum.broadcast_sub(&sum_square)?.sum_keepdim(1)? * 0.5)?)
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

    #[test]
    fn test_fm_interaction_value() {
        // Known input: 2 samples, 2 features, 2-dim embeddings
        // stacked[0] = [[1,2], [3,4]]  -> sum=4,6  square=1,4 + 9,16=10,20
        // FM = 0.5 * ((4^2+6^2) - (10+20)) = 0.5 * (16+36 - 30) = 0.5 * 22 = 11
        let t = Tensor::new(&[[[1f32, 2f32], [3f32, 4f32]]], &Device::Cpu).unwrap();
        let result = fm_interaction(&t).unwrap();
        let v = result.to_vec2::<f32>().unwrap();
        assert!((v[0][0] - 11.0).abs() < 1e-5);
    }
}
