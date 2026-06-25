//! MMoE：多门控专家混合，每个任务独立门控组合专家输出。
use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
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
    representation_names: Vec<String>,
    output_head: Option<OutputHead>,
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
            representation_names: task_names.clone(),
            task_names,
            output_head: None,
        })
    }

    /// 构造使用原生输出契约的 MMoE。
    #[allow(clippy::too_many_arguments)]
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        shared_bottom_dims: &[usize],
        num_experts: usize,
        expert_hidden_dims: &[usize],
        expert_output_dim: usize,
        contract: &OutputContract,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let (shared_bottom, shared_output_dim) = if shared_bottom_dims.is_empty() {
            (None, embeddings.total_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("shared_bottom"),
                    embeddings.total_dim,
                    &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                output_dim,
            )
        };
        let mut experts = Vec::with_capacity(num_experts);
        for index in 0..num_experts {
            experts.push(Mlp::new(
                vb.pp(format!("expert_{index}")),
                shared_output_dim,
                expert_hidden_dims,
                expert_output_dim,
                Activation::Relu,
            )?);
        }
        let mut representation_names = Vec::new();
        for tower in &contract.graph.towers {
            if !representation_names.contains(&tower.input) {
                representation_names.push(tower.input.clone());
            }
        }
        let mut gate_linears = Vec::with_capacity(representation_names.len());
        for index in 0..representation_names.len() {
            gate_linears.push(linear(
                shared_output_dim,
                num_experts,
                vb.pp(format!("gate_{index}")),
            )?);
        }
        let representation_dims = representation_names
            .iter()
            .map(|name| (name.clone(), expert_output_dim))
            .collect();
        let output_head = OutputHead::new(contract, &representation_dims, vb.pp("output_head"))?;
        Ok(Self {
            embeddings,
            shared_bottom,
            num_experts,
            experts,
            expert_output_dim,
            gate_linears,
            task_towers: vec![],
            task_names: contract
                .graph
                .towers
                .iter()
                .map(|tower| tower.name.clone())
                .collect(),
            representation_names,
            output_head: Some(output_head),
        })
    }

    fn representations(
        &self,
        x_inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let concat = self.embeddings.forward(x_inputs)?;
        let shared_output = match &self.shared_bottom {
            Some(bottom) => bottom.forward(&concat)?,
            None => concat,
        };
        let mut expert_outs = Vec::with_capacity(self.num_experts);
        for expert in &self.experts {
            expert_outs.push(expert.forward(&shared_output)?.unsqueeze(1)?);
        }
        let experts = Tensor::cat(&expert_outs, 1)?;
        let mut result = HashMap::new();
        for (index, gate_linear) in self.gate_linears.iter().enumerate() {
            let gate_weights = candle_nn::ops::softmax(&gate_linear.forward(&shared_output)?, 1)?;
            let gated = experts.broadcast_mul(&gate_weights.unsqueeze(2)?)?.sum(1)?;
            result.insert(self.representation_names[index].clone(), gated);
        }
        Ok(result)
    }
}

impl Model for MMoE {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let representations = self.representations(x_inputs)?;
        let mut outputs = ModelOutput::new();
        for (index, name) in self.task_names.iter().enumerate() {
            let logits = self.task_towers[index].forward(&representations[name])?;
            outputs.insert_binary_logit(name.clone(), logits);
        }
        Ok(outputs)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        if let Some(head) = &self.output_head {
            return head.forward(&self.representations(x_inputs)?);
        }
        let outputs = self.forward(x_inputs)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}
