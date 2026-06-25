//! 逻辑回归基线：Embedding + Linear，无特征交互。
use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
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
    output_head: Option<OutputHead>,
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
        Ok(Self {
            embeddings,
            mlp,
            output_head: None,
        })
    }

    /// 构造使用原生输出契约的逻辑回归模型。
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        contract: &OutputContract,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let mlp = Mlp::new(
            vb.pp("mlp"),
            embeddings.total_dim,
            &[],
            1,
            Activation::None_,
        )?;
        let representation_dims = HashMap::from([("shared".to_string(), 1)]);
        let output_head = OutputHead::new(contract, &representation_dims, vb.pp("output_head"))?;
        Ok(Self {
            embeddings,
            mlp,
            output_head: Some(output_head),
        })
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        self.mlp.forward(&self.embeddings.forward(x_inputs)?)
    }
}

impl Model for LogisticRegression {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let logits = self.shared(x_inputs)?;
        let mut outputs = ModelOutput::new();
        outputs.insert_binary_logit("pred", logits);
        Ok(outputs)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        if let Some(head) = &self.output_head {
            return head.forward(&HashMap::from([(
                "shared".to_string(),
                self.shared(x_inputs)?,
            )]));
        }
        let outputs = self.forward(x_inputs)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}
