//! ESMM：全量空间多任务模型（5 塔），点击条件乘积链消除 SSB。
use super::Model;
use crate::layers::embedding::FeatureEmbeddings;
use crate::layers::mlp::Mlp;
use crate::layers::towers::{Activation, TaskTower, TowerConfig};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// ESMM 5-tower model.
///
/// 概率关系:
///   is_click = is_click_detail ∨ is_click_stock
///   is_cvr ← is_click (转化在点击后发生)
///   stay_time ← is_click_detail (阅读仅在点详情后发生)
///
///   P(detail) = σ(click) × σ(detail)
///   P(stock)  = σ(click) × σ(stock)
///   P(cvr)    = σ(click) × σ(cvr)
///   P(stay)   = σ(detail) × σ(stay)
pub struct ESMM {
    embeddings: FeatureEmbeddings,
    shared_bottom: Option<Mlp>,
    click_tower: TaskTower,
    cvr_tower: TaskTower,
    detail_tower: TaskTower,
    stock_tower: TaskTower,
    stay_tower: TaskTower,
}

impl ESMM {
    pub fn new(
        vb: VarBuilder,
        features: &[(String, usize, usize)],
        shared_bottom_dims: &[usize],
        click_hidden_dims: &[usize],
        cvr_hidden_dims: &[usize],
        detail_hidden_dims: &[usize],
        stock_hidden_dims: &[usize],
        stay_hidden_dims: &[usize],
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let (shared_bottom, shared_output_dim) = if shared_bottom_dims.is_empty() {
            (None, embeddings.total_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("shared_bottom"),
                embeddings.total_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), output_dim)
        };
        let mk_tower = |name: &str, dims: &[usize], vb: VarBuilder| -> Result<TaskTower> {
            TaskTower::new(
                &TowerConfig {
                    name: name.into(),
                    hidden_dims: dims.to_vec(),
                    output_dim: 1,
                    activation: Activation::Relu,
                },
                shared_output_dim,
                vb,
            )
        };
        let click_tower = mk_tower("click", click_hidden_dims, vb.pp("click_tower"))?;
        let cvr_tower = mk_tower("cvr", cvr_hidden_dims, vb.pp("cvr_tower"))?;
        let detail_tower = mk_tower("detail", detail_hidden_dims, vb.pp("detail_tower"))?;
        let stock_tower = mk_tower("stock", stock_hidden_dims, vb.pp("stock_tower"))?;
        let stay_tower = mk_tower("stay", stay_hidden_dims, vb.pp("stay_tower"))?;
        Ok(Self {
            embeddings,
            shared_bottom,
            click_tower,
            cvr_tower,
            detail_tower,
            stock_tower,
            stay_tower,
        })
    }
}

impl Model for ESMM {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let concat = self.embeddings.forward(x_inputs)?;
        let shared_output = match &self.shared_bottom {
            Some(b) => b.forward(&concat)?,
            None => concat,
        };
        let click_logits = self.click_tower.forward(&shared_output)?;
        let cvr_logits = self.cvr_tower.forward(&shared_output)?;
        let detail_logits = self.detail_tower.forward(&shared_output)?;
        let stock_logits = self.stock_tower.forward(&shared_output)?;
        let stay_logits = self.stay_tower.forward(&shared_output)?;

        let click_prob = candle_nn::ops::sigmoid(&click_logits)?;
        let detail_prob = candle_nn::ops::sigmoid(&detail_logits)?;

        let mut outputs = HashMap::new();
        let ctcvr = click_prob.mul(&candle_nn::ops::sigmoid(&cvr_logits)?)?;
        let ctdetail = click_prob.mul(&detail_prob)?;
        let ctstock = click_prob.mul(&candle_nn::ops::sigmoid(&stock_logits)?)?;
        let ctstay = detail_prob.mul(&candle_nn::ops::sigmoid(&stay_logits)?)?;

        outputs.insert("click".into(), click_logits);
        outputs.insert("cvr".into(), cvr_logits);
        outputs.insert("detail".into(), detail_logits);
        outputs.insert("stock".into(), stock_logits);
        outputs.insert("stay".into(), stay_logits);
        outputs.insert("ctcvr".into(), ctcvr);
        outputs.insert("ctdetail".into(), ctdetail);
        outputs.insert("ctstock".into(), ctstock);
        outputs.insert("ctstay".into(), ctstay);
        Ok(outputs)
    }
}
