//! PEPNet: Parameter and Embedding Personalized Network.
//!
//! Architecture:
//!   FeatureEmbeddings → EPNet (per-domain gates) → Deep MLP → shared_bottom → PPNet gate → towers

mod build;
mod construct;
mod gate;
mod model;
mod tower;

use super::output_contract::{ContractPublicOutput, ContractRelation};
use crate::layers::embedding::FeatureEmbeddings;
use crate::layers::mlp::Mlp;

use self::gate::GateNu;
use self::tower::PersonalizedTower;

struct ContractMode {
    relations: Vec<ContractRelation>,
    outputs: Vec<ContractPublicOutput>,
}

#[allow(dead_code)]
struct DomainInfo {
    name: String,
    feature_indices: Vec<usize>,
    prior_indices: Vec<usize>,
    dim: usize,
}

/// PEPNet model with per-domain EPNet gates and per-task PPNet tower gates.
pub struct PEPNet {
    embeddings: FeatureEmbeddings,
    ep_prior_projs: Vec<candle_nn::Linear>,
    epnet_gates: Vec<GateNu>,
    domains: Vec<DomainInfo>,
    pp_prior_proj: candle_nn::Linear,
    pp_prior_indices: Vec<usize>,
    epnet_gate: Option<GateNu>,
    ep_prior_proj: Option<candle_nn::Linear>,
    ep_prior_indices: Vec<usize>,
    deep: Option<Mlp>,
    shared_bottom: Option<Mlp>,
    towers: Vec<PersonalizedTower>,
    relations: Vec<crate::layers::towers::TaskRelation>,
    contract_mode: Option<ContractMode>,
}
