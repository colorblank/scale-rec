use super::gate::GateNu;
use super::{DomainInfo, PEPNet};
use crate::layers::embedding::FeatureSpec;
use candle_core::Result;
use candle_nn::{linear_no_bias, VarBuilder};
use std::collections::HashMap;

impl PEPNet {
    pub(super) fn build_gate(vb: VarBuilder, prior_dim: usize, gate_dim: usize) -> Result<GateNu> {
        GateNu::new(vb, prior_dim, prior_dim, gate_dim)
    }

    pub(super) fn prior_indices(features: &[FeatureSpec], names: &[String]) -> Result<Vec<usize>> {
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

    pub(super) fn build_domains(
        features: &[FeatureSpec],
        domains: &[serde_yaml::Value],
        vb: VarBuilder,
        prior_dim: usize,
    ) -> Result<(Vec<DomainInfo>, Vec<candle_nn::Linear>, Vec<GateNu>)> {
        let feat_name_to_idx: HashMap<&str, usize> = features
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.as_str(), i))
            .collect();
        let feat_name_to_dim: HashMap<&str, usize> =
            features.iter().map(|f| (f.name.as_str(), f.embed_dim)).collect();

        let mut all_assigned = vec![false; features.len()];
        let mut infos = Vec::with_capacity(domains.len());
        let mut prior_projs = Vec::with_capacity(domains.len());
        let mut gates = Vec::with_capacity(domains.len());

        for d in domains.iter() {
            let name = d["name"].as_str().ok_or_else(|| {
                candle_core::Error::Msg("PEPNet domain missing 'name'".into())
            })?;
            let feat_names: Vec<&str> = d["features"]
                .as_sequence()
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "PEPNet domain '{}' missing 'features'",
                        name
                    ))
                })?
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();

            let mut feature_indices = Vec::with_capacity(feat_names.len());
            let mut dim = 0usize;
            for &fn_ in &feat_names {
                let &idx = feat_name_to_idx.get(fn_).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "PEPNet domain '{}' unknown feature '{}'",
                        name, fn_
                    ))
                })?;
                all_assigned[idx] = true;
                feature_indices.push(idx);
                dim += feat_name_to_dim.get(fn_).unwrap_or(&0);
            }

            let prior_names: Vec<&str> = d
                .get("ep_prior_features")
                .and_then(|v| v.as_sequence())
                .map(|seq| seq.iter().map(|v| v.as_str().unwrap()).collect())
                .unwrap_or_else(|| feat_names.clone());

            let mut prior_indices = Vec::with_capacity(prior_names.len());
            for &fn_ in &prior_names {
                let &idx = feat_name_to_idx.get(fn_).ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "PEPNet domain '{}' unknown ep_prior_feature '{}'",
                        name, fn_
                    ))
                })?;
                prior_indices.push(idx);
            }

            prior_projs.push(linear_no_bias(
                prior_indices.len(),
                prior_dim,
                vb.pp("ep_prior_projs").pp(name),
            )?);
            gates.push(Self::build_gate(
                vb.pp("epnet_gates").pp(name),
                prior_dim,
                dim,
            )?);
            infos.push(DomainInfo {
                name: name.to_string(),
                feature_indices,
                prior_indices,
                dim,
            });
        }

        let unassigned: Vec<String> = all_assigned
            .iter()
            .enumerate()
            .filter(|(_, &a)| !a)
            .map(|(i, _)| features[i].name.clone())
            .collect();
        if !unassigned.is_empty() {
            return Err(candle_core::Error::Msg(format!(
                "PEPNet features not assigned to any domain: {:?}",
                unassigned
            )));
        }
        Ok((infos, prior_projs, gates))
    }
}
