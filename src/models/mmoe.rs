//! MMoE：多门控专家混合，每个任务独立门控组合专家输出。
use super::{Model, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::Activation;
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::collections::HashMap;

/// MMoE 模型 (Ma et al., 2018)。
///
/// 多门控专家混合：共享底层 → N 个并行专家 → 每任务独立门控 (softmax) → 任务专属塔。
pub struct MMoE {
    embeddings: FeatureEmbeddings,
    shared_bottom: Option<Mlp>,
    /// 专家数量。
    pub num_experts: usize,
    experts: Vec<Mlp>,
    /// 每个专家的输出维度。
    pub expert_output_dim: usize,
    gate_linears: Vec<Linear>,
    task_towers: Vec<Mlp>,
    task_names: Vec<String>,
}

impl MMoE {
    #[allow(clippy::too_many_arguments)]
    /// 构造 MMoE 模型。
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        shared_bottom_dims: &[usize],
        num_experts: usize,
        expert_hidden_dims: &[usize],
        expert_output_dim: usize,
        task_configs: &[(String, Vec<usize>)],
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
        let mut experts = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            experts.push(Mlp::new(
                vb.pp(format!("expert_{}", e)),
                shared_output_dim,
                expert_hidden_dims,
                expert_output_dim,
                Activation::Relu,
            )?);
        }
        let num_tasks = task_configs.len();
        let mut gate_linears = Vec::with_capacity(num_tasks);
        for t in 0..num_tasks {
            gate_linears.push(linear(
                shared_output_dim,
                num_experts,
                vb.pp(format!("gate_{}", t)),
            )?);
        }
        let mut task_towers = Vec::with_capacity(num_tasks);
        let mut task_names = Vec::with_capacity(num_tasks);
        for (t, (name, tower_dims)) in task_configs.iter().enumerate() {
            task_names.push(name.clone());
            task_towers.push(Mlp::new(
                vb.pp(format!("task_{}_tower", t)),
                expert_output_dim,
                tower_dims,
                1,
                Activation::Relu,
            )?);
        }
        Ok(Self {
            embeddings,
            shared_bottom,
            num_experts,
            experts,
            expert_output_dim,
            gate_linears,
            task_towers,
            task_names,
        })
    }
}

impl Model for MMoE {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        let concat = self.embeddings.forward(x_inputs)?;
        let shared_output = match &self.shared_bottom {
            Some(b) => b.forward(&concat)?,
            None => concat,
        };
        let mut expert_outs = Vec::with_capacity(self.num_experts);
        for expert in &self.experts {
            expert_outs.push(expert.forward(&shared_output)?.unsqueeze(1)?);
        }
        let experts = Tensor::cat(&expert_outs, 1)?;
        let mut outputs = ModelOutput::new();
        for (t, gate_linear) in self.gate_linears.iter().enumerate() {
            let gate_weights = candle_nn::ops::softmax(&gate_linear.forward(&shared_output)?, 1)?;
            let gated_output = experts.broadcast_mul(&gate_weights.unsqueeze(2)?)?.sum(1)?;
            let logits = self.task_towers[t].forward(&gated_output)?;
            outputs.insert_binary_logit(self.task_names[t].clone(), logits);
        }
        Ok(outputs)
    }
}
