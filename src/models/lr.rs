//! 逻辑回归基线：Embedding + Linear，无特征交互。
use super::{Model, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// 逻辑回归基线模型。
///
/// Embedding → Concat → Linear(no activation) → logit。
/// 消融实验中作为无特征交互的对照组。
pub struct LogisticRegression {
    embeddings: FeatureEmbeddings,
    mlp: Mlp,
}

impl LogisticRegression {
    /// 构造逻辑回归模型。
    pub fn new(vb: VarBuilder, features: &[FeatureSpec]) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let mlp = Mlp::new(
            vb.pp("mlp"),
            embeddings.total_dim,
            &[],
            1,
            Activation::None_,
        )?;
        Ok(Self { embeddings, mlp })
    }
}

impl Model for LogisticRegression {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        let concat = self.embeddings.forward(x_inputs)?;
        let logits = self.mlp.forward(&concat)?;
        let mut outputs = ModelOutput::new();
        outputs.insert_binary_logit("pred", logits);
        Ok(outputs)
    }
}
