use super::tower::PersonalizedTower;
use super::{ContractMode, PEPNet};
use crate::layers::embedding::FeatureSpec;
use crate::layers::mlp::Mlp;
use crate::layers::towers::{Activation, MultiTaskConfig, TowerConfig};
use crate::models::output_contract::OutputContract;
use crate::models::output_head::{
    activation as contract_activation, output_kind as contract_output_kind,
};
use candle_core::Result;
use candle_nn::{linear_no_bias, VarBuilder};
use std::collections::HashMap;

impl PEPNet {
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
        domains: &[serde_yaml::Value],
    ) -> Result<Self> {
        let embeddings =
            crate::layers::embedding::FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;
        let pp_prior_indices = Self::prior_indices(features, pp_prior_features)?;
        let pp_prior_proj =
            linear_no_bias(pp_prior_indices.len(), prior_dim, vb.pp("pp_prior_proj"))?;

        let (domain_infos, ep_prior_projs, epnet_gates) = if !domains.is_empty() {
            Self::build_domains(features, domains, vb.clone(), prior_dim)?
        } else {
            (vec![], vec![], vec![])
        };

        let (ep_prior_proj, epnet_gate, ep_prior_indices) = if domains.is_empty() {
            let indices = Self::prior_indices(features, ep_prior_features)?;
            let proj = linear_no_bias(indices.len(), prior_dim, vb.pp("ep_prior_proj"))?;
            let gate = Self::build_gate(vb.pp("epnet_gate"), prior_dim, total_dim)?;
            (Some(proj), Some(gate), indices)
        } else {
            (None, None, vec![])
        };

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

        let pp_context_dim = prior_dim + total_dim;
        let mut towers = Vec::with_capacity(task_config.towers.len());
        for tower_config in &task_config.towers {
            towers.push(PersonalizedTower::new(
                tower_config,
                shared_dim,
                pp_context_dim,
                vb.pp(format!("{}_tower", tower_config.name)),
            )?);
        }

        Ok(Self {
            embeddings,
            ep_prior_projs,
            epnet_gates,
            domains: domain_infos,
            pp_prior_indices,
            pp_prior_proj,
            ep_prior_proj,
            epnet_gate,
            ep_prior_indices,
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
        domains: &[serde_yaml::Value],
    ) -> Result<Self> {
        let embeddings =
            crate::layers::embedding::FeatureEmbeddings::new(vb.pp("embeddings"), features)?;
        let total_dim = embeddings.total_dim;
        let pp_prior_indices = Self::prior_indices(features, pp_prior_features)?;
        let pp_prior_proj =
            linear_no_bias(pp_prior_indices.len(), prior_dim, vb.pp("pp_prior_proj"))?;

        let (domain_infos, ep_prior_projs, epnet_gates) = if !domains.is_empty() {
            Self::build_domains(features, domains, vb.clone(), prior_dim)?
        } else {
            (vec![], vec![], vec![])
        };

        let (ep_prior_proj, epnet_gate, ep_prior_indices) = if domains.is_empty() {
            let indices = Self::prior_indices(features, ep_prior_features)?;
            let proj = linear_no_bias(indices.len(), prior_dim, vb.pp("ep_prior_proj"))?;
            let gate = Self::build_gate(vb.pp("epnet_gate"), prior_dim, total_dim)?;
            (Some(proj), Some(gate), indices)
        } else {
            (None, None, vec![])
        };

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

        let validated = contract
            .validate(None)
            .map_err(|e| candle_core::Error::Msg(format!("validate output contract: {e}")))?;
        let pp_context_dim = prior_dim + total_dim;
        let mut towers = Vec::with_capacity(contract.graph.towers.len());
        for tower in &contract.graph.towers {
            if tower.input != "shared" {
                candle_core::bail!("PEPNet output_contract towers must use input='shared'");
            }
            towers.push(PersonalizedTower::new(
                &TowerConfig {
                    name: tower.name.clone(),
                    hidden_dims: tower.hidden_dims.clone(),
                    output_dim: 1,
                    activation: contract_activation(&tower.activation)?,
                    output_kind: contract_output_kind(tower.kind),
                },
                shared_dim,
                pp_context_dim,
                vb.pp("output_towers").pp(tower.name.clone()),
            )?);
        }
        let relations_by_name: HashMap<&str, &crate::models::output_contract::ContractRelation> =
            contract
                .graph
                .relations
                .iter()
                .map(|r| (r.name.as_str(), r))
                .collect();
        let contract_relations: Vec<crate::models::output_contract::ContractRelation> = validated
            .relation_order
            .iter()
            .map(|name| relations_by_name[name.as_str()].clone())
            .collect();

        Ok(Self {
            embeddings,
            ep_prior_projs,
            epnet_gates,
            domains: domain_infos,
            pp_prior_indices,
            pp_prior_proj,
            ep_prior_proj,
            epnet_gate,
            ep_prior_indices,
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
}
