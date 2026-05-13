use candle_core::{Result, Tensor};

pub fn fm_interaction(stacked: &Tensor) -> Result<Tensor> {
    let sum_square = stacked.sqr()?.sum(1)?;
    let square_sum = stacked.sum(1)?.sqr()?;
    Ok((sum_square.broadcast_sub(&square_sum)?.sum_keepdim(1)? * 0.5)?)
}
