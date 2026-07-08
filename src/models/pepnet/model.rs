use super::PEPNet;
use crate::layers::towers::apply_relation;
use crate::models::output_head::execute_relation as execute_contract_relation;
use crate::models::{Model, ModelExecution, ModelOutput};
use candle_core::{Module, Result, Tensor};
use std::collections::HashMap;

impl PEPNet {
    fn prior_raw(stacked: &[Tensor], indices: &[usize]) -> Result<Tensor> {
        let mut prior_parts = Vec::with_capacity(indices.len());
        for &index in indices {
            let pooled = stacked[index].mean_keepdim(2)?;
            prior_parts.push(pooled.squeeze(2)?);
        }
        Tensor::cat(&prior_parts, 1)
    }

    fn shared(&self, x_inputs: &HashMap<String, Tensor>) -> Result<(Tensor, Tensor)> {
        let stacked = self.embeddings.forward_stacked(x_inputs)?;

        let pp_prior = self
            .pp_prior_proj
            .forward(&Self::prior_raw(&stacked, &self.pp_prior_indices)?)?;

        let gated = if !self.domains.is_empty() {
            let mut gated_parts = Vec::with_capacity(self.domains.len());
            for (di, info) in self.domains.iter().enumerate() {
                let mut domain_embs = Vec::with_capacity(info.feature_indices.len());
                for &idx in &info.feature_indices {
                    domain_embs.push(stacked[idx].squeeze(1)?);
                }
                let domain_concat = Tensor::cat(&domain_embs, 1)?;
                let domain_prior = Self::prior_raw(&stacked, &info.prior_indices)?;
                let domain_prior_proj = self.ep_prior_projs[di].forward(&domain_prior)?;
                let domain_scale = self.epnet_gates[di].forward(&domain_prior_proj)?;
                gated_parts.push(domain_concat.broadcast_mul(&domain_scale)?);
            }
            Tensor::cat(&gated_parts, 1)?
        } else {
            let ep_prior = self
                .ep_prior_proj
                .as_ref()
                .unwrap()
                .forward(&Self::prior_raw(&stacked, &self.ep_prior_indices)?)?;
            let epnet_scale = self.epnet_gate.as_ref().unwrap().forward(&ep_prior)?;
            let dense_concat = Tensor::cat(
                &stacked
                    .iter()
                    .map(|e| e.squeeze(1))
                    .collect::<Result<Vec<_>>>()?,
                1,
            )?;
            dense_concat.broadcast_mul(&epnet_scale)?
        };

        let pp_context = Tensor::cat(&[&pp_prior, &gated], 1)?;

        let mut shared = match &self.deep {
            Some(deep) => deep.forward(&gated)?,
            None => gated,
        };
        if let Some(shared_bottom) = &self.shared_bottom {
            shared = shared_bottom.forward(&shared)?;
        }
        Ok((shared, pp_context))
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
