//! ESMM：全量空间多任务模型，CTR·CVR 乘积链消除 SSB。
use super::Model;
use crate::layers::embedding::FeatureEmbeddings;
use crate::layers::mlp::Mlp;
use crate::layers::towers::{Activation, TaskTower, TowerConfig};
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// ESMM 模型 (Ma et al., 2018)。
///
/// 全量空间多任务：CTR 塔 + CVR 塔，CTCVR = σ(CTR) × σ(CVR)。
/// 在全量曝光上训练，消除 CVR 样本选择偏差 (SSB)。
pub struct ESMM {
    embeddings: FeatureEmbeddings,
    shared_bottom: Option<Mlp>,
    ctr_tower: TaskTower,
    cvr_tower: TaskTower,
}

impl ESMM {
    pub fn new(
        vb: VarBuilder,
        features: &[(String, usize, usize)],
        shared_bottom_dims: &[usize],
        ctr_hidden_dims: &[usize],
        cvr_hidden_dims: &[usize],
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
        let ctr_tower = TaskTower::new(
            &TowerConfig {
                name: "ctr".into(),
                hidden_dims: ctr_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
            },
            shared_output_dim,
            vb.pp("ctr_tower"),
        )?;
        let cvr_tower = TaskTower::new(
            &TowerConfig {
                name: "cvr".into(),
                hidden_dims: cvr_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
            },
            shared_output_dim,
            vb.pp("cvr_tower"),
        )?;
        Ok(Self {
            embeddings,
            shared_bottom,
            ctr_tower,
            cvr_tower,
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
        let ctr_logits = self.ctr_tower.forward(&shared_output)?;
        let cvr_logits = self.cvr_tower.forward(&shared_output)?;
        let ctr_prob = candle_nn::ops::sigmoid(&ctr_logits)?;
        let cvr_prob = candle_nn::ops::sigmoid(&cvr_logits)?;
        let ctcvr = ctr_prob.mul(&cvr_prob)?;
        let mut outputs = HashMap::new();
        outputs.insert("ctr".to_string(), ctr_logits);
        outputs.insert("cvr".to_string(), cvr_logits);
        outputs.insert("ctcvr".to_string(), ctcvr);
        Ok(outputs)
    }
}
