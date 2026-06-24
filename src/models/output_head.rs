//! Contract-driven task towers, relation graph execution and output projection.

use std::collections::HashMap;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::layers::towers::{Activation, TaskTower, TowerConfig};
use crate::models::output_contract::{
    ContractNodeKind, ContractPublicOutput, ContractRelation, ContractRelationOp, OutputContract,
};
use crate::models::{ModelExecution, ModelOutput, OutputKind};

struct TowerBinding {
    input: String,
    tower: TaskTower,
}

/// Generic output head built from an [`OutputContract`].
pub struct OutputHead {
    towers: Vec<TowerBinding>,
    relations: Vec<ContractRelation>,
    outputs: Vec<ContractPublicOutput>,
}

impl OutputHead {
    /// Build scalar towers for the named backbone representations.
    pub fn new(
        contract: &OutputContract,
        representation_dims: &HashMap<String, usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let validated = contract.validate(None).map_err(|error| {
            candle_core::Error::Msg(format!("validate output contract: {error}"))
        })?;
        let mut towers = Vec::with_capacity(contract.graph.towers.len());
        for tower in &contract.graph.towers {
            let input_dim = representation_dims.get(&tower.input).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "tower '{}' references unknown representation '{}'",
                    tower.name, tower.input
                ))
            })?;
            if *input_dim == 0 {
                return Err(candle_core::Error::Msg(format!(
                    "representation '{}' dimension must be positive, got 0",
                    tower.input
                )));
            }
            let config = TowerConfig {
                name: tower.name.clone(),
                hidden_dims: tower.hidden_dims.clone(),
                output_dim: 1,
                activation: activation(&tower.activation)?,
                output_kind: output_kind(tower.kind),
            };
            towers.push(TowerBinding {
                input: tower.input.clone(),
                tower: TaskTower::new(&config, *input_dim, vb.pp("towers").pp(tower.name.clone()))?,
            });
        }
        let relations_by_name: HashMap<&str, &ContractRelation> = contract
            .graph
            .relations
            .iter()
            .map(|relation| (relation.name.as_str(), relation))
            .collect();
        let relations = validated
            .relation_order
            .iter()
            .map(|name| relations_by_name[name.as_str()].clone())
            .collect();
        Ok(Self {
            towers,
            relations,
            outputs: contract.outputs.clone(),
        })
    }

    /// Execute all towers and relations, then project the configured public outputs.
    pub fn forward(&self, representations: &HashMap<String, Tensor>) -> Result<ModelExecution> {
        let mut nodes = ModelOutput::new();
        for binding in &self.towers {
            let input = representations.get(&binding.input).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "output head representation '{}' is missing",
                    binding.input
                ))
            })?;
            nodes.insert(
                binding.tower.name().to_string(),
                binding.tower.forward(input)?,
                binding.tower.output_kind(),
            );
        }
        for relation in &self.relations {
            let (tensor, kind) = execute_relation(relation, &nodes)?;
            nodes.insert(relation.name.clone(), tensor, kind);
        }

        let mut outputs = ModelOutput::new();
        for output in &self.outputs {
            let source = nodes.get(&output.source).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "public output '{}' source '{}' is missing",
                    output.name, output.source
                ))
            })?;
            outputs.insert(output.name.clone(), source.tensor.clone(), source.kind);
        }
        Ok(ModelExecution::new(nodes, outputs))
    }
}

fn execute_relation(
    relation: &ContractRelation,
    nodes: &ModelOutput,
) -> Result<(Tensor, OutputKind)> {
    let inputs: Vec<_> = relation
        .inputs
        .iter()
        .map(|name| {
            nodes.get(name).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "relation '{}' input '{}' is missing",
                    relation.name, name
                ))
            })
        })
        .collect::<Result<_>>()?;
    match relation.op {
        ContractRelationOp::Sigmoid => Ok((
            candle_nn::ops::sigmoid(&inputs[0].tensor)?,
            OutputKind::Probability,
        )),
        ContractRelationOp::Multiply => {
            let mut result = inputs[0].tensor.clone();
            for input in &inputs[1..] {
                result = result.mul(&input.tensor)?;
            }
            Ok((result, OutputKind::Probability))
        }
        ContractRelationOp::Add => {
            let mut result = inputs[0].tensor.clone();
            for input in &inputs[1..] {
                result = result.broadcast_add(&input.tensor)?;
            }
            Ok((result, OutputKind::Regression))
        }
        ContractRelationOp::Identity => Ok((inputs[0].tensor.clone(), inputs[0].kind)),
    }
}

fn activation(name: &str) -> Result<Activation> {
    match name {
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        "swish" => Ok(Activation::Swish),
        "gelu" => Ok(Activation::Gelu),
        "none" => Ok(Activation::None_),
        _ => Err(candle_core::Error::Msg(format!(
            "unsupported tower activation '{name}'"
        ))),
    }
}

fn output_kind(kind: ContractNodeKind) -> OutputKind {
    match kind {
        ContractNodeKind::BinaryLogit => OutputKind::BinaryLogit,
        ContractNodeKind::Probability => OutputKind::Probability,
        ContractNodeKind::Regression => OutputKind::Regression,
        ContractNodeKind::Score => OutputKind::Score,
    }
}
