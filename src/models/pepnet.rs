//! PEPNet: Parameter and Embedding Personalized Network.
//!
//! Architecture:
//!   FeatureEmbeddings → EPNet gate → Deep MLP → shared_bottom → PPNet gate → towers
//!
//! EPNet computes an element-wise gate on the concatenated embedding vector,
//! conditioned on selected personalized prior features. PPNet applies per-layer gates inside each
//! task tower using selected personalized prior features.

use super::output_contract::{ContractPublicOutput, ContractRelation, OutputContract};
use super::output_head::{
    activation as contract_activation, execute_relation as execute_contract_relation,
    output_kind as contract_output_kind,
};
use super::{Model, ModelExecution, ModelOutput, OutputKind};
use crate::layers::embedding::{FeatureEmbeddings, FeatureSpec};
use crate::layers::mlp::Mlp;
use crate::layers::towers::{
    apply_relation, Activation, MultiTaskConfig, TaskRelation, TowerConfig,
};
use candle_core::{Result, Tensor};
use candle_nn::{linear, linear_no_bias, Module, VarBuilder};
use std::collections::HashMap;

struct GateNu {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    gamma: f64,
}

impl GateNu {
    fn new(vb: VarBuilder, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        Ok(Self {
            fc1: linear(input_dim, hidden_dim, vb.pp("fc1"))?,
            fc2: linear(hidden_dim, output_dim, vb.pp("fc2"))?,
            gamma: 2.0,
        })
    }
}

impl Module for GateNu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(x)?.relu()?;
        candle_nn::ops::sigmoid(&self.fc2.forward(&hidden)?)?.affine(self.gamma, 0.0)
    }
}

struct PersonalizedTower {
    name: String,
    layers: Vec<candle_nn::Linear>,
    pp_gates: Vec<GateNu>,
    activation: Activation,
    output_kind: OutputKind,
}

impl PersonalizedTower {
    fn new(
        config: &TowerConfig,
        input_dim: usize,
        prior_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.hidden_dims.len() + 1);
        let mut pp_gates = Vec::with_capacity(config.hidden_dims.len());
        let mut in_dim = input_dim;
        for (i, &h_dim) in config.hidden_dims.iter().enumerate() {
            layers.push(linear(in_dim, h_dim, vb.pp(format!("hidden.{}", i)))?);
            pp_gates.push(GateNu::new(
                vb.pp("pp_gates").pp(i.to_string()),
                prior_dim,
                prior_dim,
                h_dim,
            )?);
            in_dim = h_dim;
        }
        layers.push(linear(
            in_dim,
            config.output_dim,
            vb.pp(format!("output.{}", config.hidden_dims.len())),
        )?);
        Ok(Self {
            name: config.name.clone(),
            layers,
            pp_gates,
            activation: config.activation.clone(),
            output_kind: config.output_kind,
        })
    }

    fn forward(&self, shared: &Tensor, prior: &Tensor) -> Result<Tensor> {
        let mut out = shared.clone();
        let last = self.layers.len() - 1;
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;
            if i < last {
                out = apply_activation(&self.activation, &out)?;
                out = out.broadcast_mul(&self.pp_gates[i].forward(prior)?)?;
            }
        }
        Ok(out)
    }
}

struct ContractMode {
    relations: Vec<ContractRelation>,
    outputs: Vec<ContractPublicOutput>,
}

/// PEPNet model with EPNet embedding gates and per-task PPNet tower gates.
pub struct PEPNet {
    embeddings: FeatureEmbeddings,
    ep_prior_indices: Vec<usize>,
    pp_prior_indices: Vec<usize>,
    /// Projects feature-wise pooled prior → fixed-dim prior representation.
    ep_prior_proj: candle_nn::Linear,
    pp_prior_proj: candle_nn::Linear,
    /// EPNet gate: prior_rep → [total_dim] sigmoid gate applied to embeddings.
    epnet_gate: GateNu,
    deep: Option<Mlp>,
    shared_bottom: Option<Mlp>,
    towers: Vec<PersonalizedTower>,
    relations: Vec<TaskRelation>,
    contract_mode: Option<ContractMode>,
}

impl PEPNet {
    fn build_gate(vb: VarBuilder, prior_dim: usize, gate_dim: usize) -> Result<GateNu> {
        GateNu::new(vb, prior_dim, prior_dim, gate_dim)
    }

    fn prior_indices(features: &[FeatureSpec], names: &[String]) -> Result<Vec<usize>> {
        if names.is_empty() {
            return Ok((0..features.len()).collect());
        }
        let mut indices = Vec::with_capacity(names.len());
        for name in names {
            let index = features
                .iter()
                .position(|feature| feature.name == *name)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "PEPNet prior feature '{}' is not embeddable",
                        name
                    ))
                })?;
            indices.push(index);
        }
        Ok(indices)
    }

    /// Build PEPNet from the legacy multi-task tower configuration.
    pub fn new(
        vb: VarBuilder,
        features: &[FeatureSpec],
        prior_dim: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        task_config: &MultiTaskConfig,
        ep_prior_features: &[String],
        pp_prior_features: &[String],
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;
        let ep_prior_indices = Self::prior_indices(features, ep_prior_features)?;
        let pp_prior_indices = Self::prior_indices(features, pp_prior_features)?;

        let ep_prior_proj =
            linear_no_bias(ep_prior_indices.len(), prior_dim, vb.pp("ep_prior_proj"))?;
        let pp_prior_proj =
            linear_no_bias(pp_prior_indices.len(), prior_dim, vb.pp("pp_prior_proj"))?;
        let epnet_gate = Self::build_gate(vb.pp("epnet_gate"), prior_dim, total_dim)?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, total_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            let mlp = Mlp::new(
                vb.pp("deep"),
                total_dim,
                &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                output_dim,
                Activation::Relu,
            )?;
            (Some(mlp), output_dim)
        };

        let (shared_bottom, shared_dim) = if shared_bottom_dims.is_empty() {
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
            towers.push(PersonalizedTower::new(
                tower_config,
                shared_dim,
                prior_dim,
                vb.pp(format!("{}_tower", tower_config.name)),
            )?);
        }

        Ok(Self {
            embeddings,
            ep_prior_indices,
            pp_prior_indices,
            ep_prior_proj,
            pp_prior_proj,
            epnet_gate,
            deep,
            shared_bottom,
            towers,
            relations: task_config.relations.clone(),
            contract_mode: None,
        })
    }

    /// Build PEPNet from a native output contract configuration.
    pub fn with_output_contract(
        vb: VarBuilder,
        features: &[FeatureSpec],
        prior_dim: usize,
        deep_hidden_dims: &[usize],
        shared_bottom_dims: &[usize],
        contract: &OutputContract,
        ep_prior_features: &[String],
        pp_prior_features: &[String],
    ) -> Result<Self> {
        let embeddings = FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;
        let ep_prior_indices = Self::prior_indices(features, ep_prior_features)?;
        let pp_prior_indices = Self::prior_indices(features, pp_prior_features)?;

        let ep_prior_proj =
            linear_no_bias(ep_prior_indices.len(), prior_dim, vb.pp("ep_prior_proj"))?;
        let pp_prior_proj =
            linear_no_bias(pp_prior_indices.len(), prior_dim, vb.pp("pp_prior_proj"))?;
        let epnet_gate = Self::build_gate(vb.pp("epnet_gate"), prior_dim, total_dim)?;

        let (deep, fusion_dim) = if deep_hidden_dims.is_empty() {
            (None, total_dim)
        } else {
            let output_dim = *deep_hidden_dims.last().unwrap();
            (
                Some(Mlp::new(
                    vb.pp("deep"),
                    total_dim,
                    &deep_hidden_dims[..deep_hidden_dims.len() - 1],
                    output_dim,
                    Activation::Relu,
                )?),
                output_dim,
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
        let validated = contract.validate(None).map_err(|error| {
            candle_core::Error::Msg(format!("validate output contract: {error}"))
        })?;
        let mut towers = Vec::with_capacity(contract.graph.towers.len());
        for tower in &contract.graph.towers {
            if tower.input != "shared" {
                candle_core::bail!("PEPNet output_contract towers must use input='shared'");
            }
            let config = TowerConfig {
                name: tower.name.clone(),
                hidden_dims: tower.hidden_dims.clone(),
                output_dim: 1,
                activation: contract_activation(&tower.activation)?,
                output_kind: contract_output_kind(tower.kind),
            };
            towers.push(PersonalizedTower::new(
                &config,
                shared_dim,
                prior_dim,
                vb.pp("output_towers").pp(tower.name.clone()),
            )?);
        }
        let relations_by_name: HashMap<&str, &ContractRelation> = contract
            .graph
            .relations
            .iter()
            .map(|relation| (relation.name.as_str(), relation))
            .collect();
        let contract_relations = validated
            .relation_order
            .iter()
            .map(|name| relations_by_name[name.as_str()].clone())
            .collect();

        Ok(Self {
            embeddings,
            ep_prior_indices,
            pp_prior_indices,
            ep_prior_proj,
            pp_prior_proj,
            epnet_gate,
            deep,
            shared_bottom,
            towers,
            relations: vec![],
            contract_mode: Some(ContractMode {
                relations: contract_relations,
                outputs: contract.outputs.clone(),
            }),
        })
    }

    fn prior_raw(stacked: &[Tensor], indices: &[usize]) -> Result<Tensor> {
        let mut prior_parts = Vec::with_capacity(indices.len());
        for &index in indices {
            let pooled = stacked[index].mean_keepdim(2)?;
            prior_parts.push(pooled.squeeze(2)?);
        }
        Tensor::cat(&prior_parts, 1)
    }

    /// Build the shared representation and PPNet prior.
    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<(Tensor, Tensor)> {
        let stacked = self.embeddings.forward_stacked(x_inputs)?;
        // stacked: Vec<[batch, 1, dim_i]>

        let ep_prior = self
            .ep_prior_proj
            .forward(&Self::prior_raw(&stacked, &self.ep_prior_indices)?)?;
        let pp_prior = self
            .pp_prior_proj
            .forward(&Self::prior_raw(&stacked, &self.pp_prior_indices)?)?;

        // EPNet gate on embeddings
        let epnet_scale = self.epnet_gate.forward(&ep_prior)?; // [batch, total_dim]
        let dense_concat = Tensor::cat(
            &stacked
                .iter()
                .map(|e| e.squeeze(1))
                .collect::<Result<Vec<_>>>()?,
            1,
        )?; // [batch, total_dim]
        let gated = dense_concat.broadcast_mul(&epnet_scale)?;

        let mut shared = match &self.deep {
            Some(deep) => deep.forward(&gated)?,
            None => gated,
        };
        if let Some(shared_bottom) = &self.shared_bottom {
            shared = shared_bottom.forward(&shared)?;
        }

        Ok((shared, pp_prior))
    }
}

impl Model for PEPNet {
    fn forward(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelOutput> {
        if self.contract_mode.is_some() {
            return Ok(self.forward_execution(x_inputs)?.outputs);
        }
        let (shared, pp_prior) = self.shared(x_inputs)?;
        let mut outputs = ModelOutput::new();
        for tower in &self.towers {
            outputs.insert(
                tower.name.clone(),
                tower.forward(&shared, &pp_prior)?,
                tower.output_kind,
            );
        }
        for relation in &self.relations {
            outputs
                .insert_probability(relation.target.clone(), apply_relation(relation, &outputs)?);
        }
        Ok(outputs)
    }

    fn forward_execution(&self, x_inputs: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        if let Some(contract) = &self.contract_mode {
            let (shared, pp_prior) = self.shared(x_inputs)?;
            let mut nodes = ModelOutput::new();
            for tower in &self.towers {
                nodes.insert(
                    tower.name.clone(),
                    tower.forward(&shared, &pp_prior)?,
                    tower.output_kind,
                );
            }
            for relation in &contract.relations {
                let (tensor, kind) = execute_contract_relation(relation, &nodes)?;
                nodes.insert(relation.name.clone(), tensor, kind);
            }
            let mut outputs = ModelOutput::new();
            for output in &contract.outputs {
                let source = nodes.get(&output.source).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "public output '{}' source '{}' is missing",
                        output.name, output.source
                    ))
                })?;
                outputs.insert(output.name.clone(), source.tensor.clone(), source.kind);
            }
            return Ok(ModelExecution::new(nodes, outputs));
        }
        let outputs = self.forward(x_inputs)?;
        Ok(ModelExecution::new(outputs.clone(), outputs))
    }
}

fn apply_activation(activation: &Activation, x: &Tensor) -> Result<Tensor> {
    match activation {
        Activation::Relu => x.relu(),
        Activation::Sigmoid => candle_nn::ops::sigmoid(x),
        Activation::Swish => {
            let sig = candle_nn::ops::sigmoid(x)?;
            x.mul(&sig)
        }
        Activation::Gelu => x.gelu(),
        Activation::None_ => Ok(x.clone()),
    }
}
