//! DeepFM：FM 一阶 + FM 二阶 + Deep MLP 的联合模型。
use super::{Model, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::fm::fm_interaction;
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// DeepFM 模型 (Guo et al., 2017)。
///
/// FM 一阶 (标量权重求和) + FM 二阶 (隐向量内积交互) + Deep MLP。
/// 最终 `logit = first_order + second_order + deep_out + global_bias`。
pub struct DeepFM {
    fm_first_embeddings: FeatureEmbeddings,
    fm_second_embeddings: FeatureEmbeddings,
    /// FM 二阶隐向量维度。
    pub fm_k: usize,
    deep_embeddings: FeatureEmbeddings,
    /// Deep 分支输入总维度。
    pub deep_total_dim: usize,
    deep_mlp: Mlp,
    global_bias: Tensor,
}

impl DeepFM {
    /// 构造 DeepFM 模型：FM 一阶/二阶 + Deep MLP 分支。
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        fm_k: usize,
        deep_hidden_dims: &[usize],
    ) -> Result<Self> {
        let fm_first_cfg: Vec<FeatureSpec> = features.iter().map(|f| f.with_dim(1)).collect();
        let fm_first_embeddings = FeatureEmbeddings::new(vb.pp("fm_first"), &fm_first_cfg)?;
        let fm_second_cfg: Vec<FeatureSpec> = features.iter().map(|f| f.with_dim(fm_k)).collect();
        let fm_second_embeddings = FeatureEmbeddings::new(vb.pp("fm_second"), &fm_second_cfg)?;
        let deep_embeddings = FeatureEmbeddings::new(vb.pp("deep"), features)?;
        let deep_mlp = Mlp::new(
            vb.pp("deep_mlp"),
            deep_embeddings.total_dim,
            deep_hidden_dims,
            1,
            Activation::Relu,
        )?;
        let global_bias = vb.get_with_hints((1,), "global_bias", candle_nn::Init::Const(0.0))?;
        let deep_total_dim = deep_embeddings.total_dim;
        Ok(Self {
            fm_first_embeddings,
            fm_second_embeddings,
            fm_k,
            deep_embeddings,
            deep_total_dim,
            deep_mlp,
            global_bias,
        })
    }
}

impl Model for DeepFM {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        let first_order = self.fm_first_embeddings.forward(x_inputs)?.sum_keepdim(1)?;
        let fm_stacked = Tensor::cat(&self.fm_second_embeddings.forward_stacked(x_inputs)?, 1)?;
        let second_order = fm_interaction(&fm_stacked)?;
        let deep_input = self.deep_embeddings.forward(x_inputs)?;
        let deep_out = self.deep_mlp.forward(&deep_input)?;
        let logits = first_order
            .broadcast_add(&second_order)?
            .broadcast_add(&deep_out)?
            .broadcast_add(&self.global_bias)?;
        let mut outputs = ModelOutput::new();
        outputs.insert_binary_logit("pred", logits);
        Ok(outputs)
    }
}
