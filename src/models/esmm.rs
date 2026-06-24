//! Configurable ESMM: shared bottom + task towers + probability relations.
use super::{Model, ModelExecution, ModelOutput, OutputKind};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::{
    apply_relation, Activation, MultiTaskConfig, RelationOp, TaskRelation, TaskTower, TowerConfig,
};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::OutputHead;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// 生成默认 5 任务 ESMM 配置 (click/cvr/detail/stock/stay)。
pub fn default_task_config(
    click_hidden_dims: &[usize],
    cvr_hidden_dims: &[usize],
    detail_hidden_dims: &[usize],
    stock_hidden_dims: &[usize],
    stay_hidden_dims: &[usize],
) -> MultiTaskConfig {
    MultiTaskConfig {
        towers: vec![
            TowerConfig {
                name: "click".into(),
                hidden_dims: click_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
            TowerConfig {
                name: "cvr".into(),
                hidden_dims: cvr_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
            TowerConfig {
                name: "detail".into(),
                hidden_dims: detail_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
            TowerConfig {
                name: "stock".into(),
                hidden_dims: stock_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
            TowerConfig {
                name: "stay".into(),
                hidden_dims: stay_hidden_dims.to_vec(),
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
        ],
        relations: vec![
            TaskRelation {
                target: "ctcvr".into(),
                sources: vec!["click".into(), "cvr".into()],
                op: RelationOp::Multiply,
            },
            TaskRelation {
                target: "ctdetail".into(),
                sources: vec!["click".into(), "detail".into()],
                op: RelationOp::Multiply,
            },
            TaskRelation {
                target: "ctstock".into(),
                sources: vec!["click".into(), "stock".into()],
                op: RelationOp::Multiply,
            },
            TaskRelation {
                target: "ctstay".into(),
                sources: vec!["detail".into(), "stay".into()],
                op: RelationOp::Multiply,
            },
        ],
    }
}

/// Entire-space multi-task model with configurable towers and relations.
pub struct ESMM {
    embeddings: FeatureEmbeddings,
    shared_bottom: Option<Mlp>,
    towers: Vec<(String, TaskTower)>,
    relations: Vec<TaskRelation>,
}

impl ESMM {
    #[allow(clippy::too_many_arguments)]
    /// 使用默认任务配置构造 ESMM。
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
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
        Self::with_task_config(vb, features, shared_bottom_dims, &task_config)
    }

    /// 使用自定义 MultiTaskConfig 构造 ESMM。
    pub fn with_task_config(
        vb: VarBuilder,
        features: &[FeatureSpec],
        shared_bottom_dims: &[usize],
        task_config: &MultiTaskConfig,
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
        let mut towers = Vec::with_capacity(task_config.towers.len());
        for tower_config in &task_config.towers {
            towers.push((
                tower_config.name.clone(),
                TaskTower::new(
                    tower_config,
                    shared_output_dim,
                    vb.pp(format!("{}_tower", tower_config.name)),
                )?,
            ));
        }
        Ok(Self {
            embeddings,
            shared_bottom,
            towers,
            relations: task_config.relations.clone(),
        })
    }
}

impl Model for ESMM {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        let concat = self.embeddings.forward(x_inputs)?;
        let shared_output = match &self.shared_bottom {
            Some(b) => b.forward(&concat)?,
            None => concat,
        };
        let mut outputs = ModelOutput::new();
        for (name, tower) in &self.towers {
            outputs.insert_binary_logit(name.clone(), tower.forward(&shared_output)?);
        }
        for relation in &self.relations {
            outputs
                .insert_probability(relation.target.clone(), apply_relation(relation, &outputs)?);
        }
        Ok(outputs)
    }
}

/// Standard ESMM backbone using the generic output contract execution path.
pub struct ContractEsmm {
    embeddings: FeatureEmbeddings,
    shared_bottom: Option<Mlp>,
    output_head: OutputHead,
}

impl ContractEsmm {
    /// Build a shared-bottom ESMM with contract-defined towers and relations.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        shared_bottom_dims: &[usize],
        contract: &OutputContract,
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let (shared_bottom, shared_output_dim) = if shared_bottom_dims.is_empty() {
            (None, embeddings.total_dim)
        } else {
            let output_dim = *shared_bottom_dims.last().ok_or_else(|| {
                candle_core::Error::Msg("shared_bottom_dims must not be empty".into())
            })?;
            let mlp = Mlp::new(
                vb.pp("shared_bottom"),
                embeddings.total_dim,
                &shared_bottom_dims[..shared_bottom_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), output_dim)
        };
        let representation_dims = HashMap::from([("shared".to_string(), shared_output_dim)]);
        let output_head = OutputHead::new(contract, &representation_dims, vb.pp("output_head"))?;
        Ok(Self {
            embeddings,
            shared_bottom,
            output_head,
        })
    }

    fn representations(
        &self,
        x_inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let concat = self.embeddings.forward(x_inputs)?;
        let shared = match &self.shared_bottom {
            Some(bottom) => bottom.forward(&concat)?,
            None => concat,
        };
        Ok(HashMap::from([("shared".to_string(), shared)]))
    }
}

impl Model for ContractEsmm {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        Ok(self.forward_execution(x_inputs)?.outputs)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        self.output_head.forward(&self.representations(x_inputs)?)
    }
}
