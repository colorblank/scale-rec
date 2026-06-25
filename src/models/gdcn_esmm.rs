//! GDCN + ESMM: gated cross network shared representation with ESMM task towers.
use super::output_contract::OutputContract;
use super::output_head::OutputHead;
use super::{Model, ModelExecution, ModelOutput};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::gdcn::GatedCrossNetwork;
use crate::layers::mlp::Mlp;
use crate::layers::towers::{apply_relation, Activation, MultiTaskConfig, TaskRelation, TaskTower};
use crate::models::esmm::default_task_config;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use std::collections::HashMap;

/// ESMM variant using a gated cross network plus optional deep branch.
pub struct GDCNESMM {
    embeddings: FeatureEmbeddings,
    cross: GatedCrossNetwork,
    deep: Option<Mlp>,
    shared_bottom: Option<Mlp>,
    towers: Vec<(String, TaskTower)>,
    relations: Vec<TaskRelation>,
    output_head: Option<OutputHead>,
}

impl GDCNESMM {
    #[allow(clippy::too_many_arguments)]
    /// 使用默认任务配置构造 GDCNESMM。
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        cross_layers: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        click_hidden_dims: &[usize],
        cvr_hidden_dims: &[usize],
        detail_hidden_dims: &[usize],
        stock_hidden_dims: &[usize],
        stay_hidden_dims: &[usize],
    ) -> Result<Self> {
        let task_config = default_task_config(
            click_hidden_dims,
            cvr_hidden_dims,
            detail_hidden_dims,
            stock_hidden_dims,
            stay_hidden_dims,
        );
        Self::with_task_config(
            vb,
            features,
            cross_layers,
            deep_hidden_dims,
            shared_bottom_dims,
            &task_config,
        )
    }

    /// 使用自定义 MultiTaskConfig 构造 GDCNESMM。
    pub fn with_task_config(
        vb: VarBuilder,
        features: &[FeatureSpec],
        cross_layers: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        task_config: &MultiTaskConfig,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let input_dim = embeddings.total_dim;
        let cross = GatedCrossNetwork::new(vb.pp("cross"), input_dim, cross_layers)?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, input_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("deep"),
                input_dim,
                &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), input_dim + output_dim)
        };

        let (shared_bottom, tower_input_dim) = if shared_bottom_dims.is_empty() {
            (None, fusion_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("shared_bottom"),
                fusion_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), output_dim)
        };

        let mut towers = Vec::with_capacity(task_config.towers.len());
        for tower_config in &task_config.towers {
            towers.push((
                tower_config.name.clone(),
                TaskTower::new(
                    tower_config,
                    tower_input_dim,
                    vb.pp(format!("{}_tower", tower_config.name)),
                )?,
            ));
        }

        Ok(Self {
            embeddings,
            cross,
            deep,
            shared_bottom,
            towers,
            relations: task_config.relations.clone(),
            output_head: None,
        })
    }

    /// 使用原生输出契约构造 GDCNESMM。
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        cross_layers: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        contract: &OutputContract,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let input_dim = embeddings.total_dim;
        let cross = GatedCrossNetwork::new(vb.pp("cross"), input_dim, cross_layers)?;
        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, input_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("deep"),
                    input_dim,
                    &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                input_dim + output_dim,
            )
        };
        let (shared_bottom, shared_dim) = if shared_bottom_dims.is_empty() {
            (None, fusion_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("shared_bottom"),
                    fusion_dim,
                    &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                output_dim,
            )
        };
        let output_head = OutputHead::new(
            contract,
            &HashMap::from([("shared".to_string(), shared_dim)]),
            vb.pp("output_head"),
        )?;
        Ok(Self {
            embeddings,
            cross,
            deep,
            shared_bottom,
            towers: vec![],
            relations: vec![],
            output_head: Some(output_head),
        })
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<Tensor> {
        let dense = self.embeddings.forward(x_inputs)?;
        let cross_out = self.cross.forward(&dense)?;
        let mut shared = match &self.deep {
            Some(deep) => Tensor::cat(&[cross_out, deep.forward(&dense)?], 1)?,
            None => cross_out,
        };
        if let Some(shared_bottom) = &self.shared_bottom {
            shared = shared_bottom.forward(&shared)?;
        }
        Ok(shared)
    }
}

impl Model for GDCNESMM {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.output_head.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let shared = self.shared(x_inputs)?;
        let mut outputs = ModelOutput::new();
        for (name, tower) in &self.towers {
            outputs.insert_binary_logit(name.clone(), tower.forward(&shared)?);
        }
        for relation in &self.relations {
            outputs
                .insert_probability(relation.target.clone(), apply_relation(relation, &outputs)?);
        }
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
