//! Versioned output-contract schema and semantic validation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Output graph node semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractNodeKind {
    /// Binary classification logit.
    BinaryLogit,
    /// Probability in `[0, 1]`.
    Probability,
    /// Continuous regression value.
    Regression,
    /// Uncalibrated ranking score.
    Score,
}

/// Source metadata required for cross-config validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceContract {
    /// Source name.
    pub name: String,
    /// Source role.
    pub role: String,
    /// Whether the source declares a default value.
    #[serde(default)]
    pub has_default: bool,
}

/// A scalar prediction tower.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractTower {
    /// Graph node name.
    pub name: String,
    /// Backbone representation name.
    #[serde(default = "default_shared")]
    pub input: String,
    /// Raw tower output semantics.
    pub kind: ContractNodeKind,
    /// Hidden layer dimensions.
    #[serde(default)]
    pub hidden_dims: Vec<usize>,
    /// Hidden layer activation.
    #[serde(default = "default_activation")]
    pub activation: String,
}

fn default_shared() -> String {
    "shared".into()
}

fn default_activation() -> String {
    "relu".into()
}

/// Supported parameter-free relation operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContractRelationOp {
    /// Convert a binary logit to probability.
    Sigmoid,
    /// Multiply two or more probabilities.
    Multiply,
    /// Add two or more regression values.
    Add,
    /// Preserve an input value and kind.
    Identity,
}

/// A relation DAG node.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractRelation {
    /// Graph node name.
    pub name: String,
    /// Relation operation.
    pub op: ContractRelationOp,
    /// Input graph nodes.
    pub inputs: Vec<String>,
}

/// Output graph configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputGraphContract {
    /// Scalar tower nodes.
    pub towers: Vec<ContractTower>,
    /// Parameter-free relation nodes.
    pub relations: Vec<ContractRelation>,
}

/// Structured sample mask.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractMask {
    /// Raw data source name.
    pub source: String,
    /// Comparison operation.
    pub op: String,
    /// Comparison value when required.
    #[serde(default)]
    pub value: Option<serde_yaml::Value>,
}

/// Loss configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractLoss {
    /// Registered loss type.
    #[serde(rename = "type")]
    pub loss_type: String,
    /// Sample reduction.
    #[serde(default = "default_reduction")]
    pub reduction: String,
    /// Probability clamp epsilon.
    #[serde(default)]
    pub epsilon: Option<f64>,
    /// Positive sample weight.
    #[serde(default)]
    pub pos_weight: Option<f64>,
    /// Huber delta.
    #[serde(default)]
    pub delta: Option<f64>,
    /// Focal loss alpha.
    #[serde(default)]
    pub alpha: Option<f64>,
    /// Focal loss gamma.
    #[serde(default)]
    pub gamma: Option<f64>,
}

fn default_reduction() -> String {
    "mean".into()
}

/// Training objective configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractObjective {
    /// Objective name.
    pub name: String,
    /// Referenced graph node.
    pub source: String,
    /// Referenced label source.
    pub label: String,
    /// Registered loss configuration.
    pub loss: ContractLoss,
    /// Objective weight.
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Optional sample mask.
    #[serde(default)]
    pub mask: Option<ContractMask>,
}

fn default_weight() -> f64 {
    1.0
}

/// Metric configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractMetric {
    /// Metric name.
    pub name: String,
    /// Referenced graph node.
    pub source: String,
    /// Referenced label source.
    pub label: String,
    /// Registered metric type.
    #[serde(rename = "type")]
    pub metric_type: String,
    /// Optional sample mask.
    #[serde(default)]
    pub mask: Option<ContractMask>,
}

/// Public serving output projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractPublicOutput {
    /// Stable public output name.
    pub name: String,
    /// Referenced graph node.
    pub source: String,
}

/// Native output contract version 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputContract {
    /// Contract schema version.
    pub version: u32,
    /// Tower and relation graph.
    pub graph: OutputGraphContract,
    /// Training objectives.
    pub objectives: Vec<ContractObjective>,
    /// Evaluation metrics.
    pub metrics: Vec<ContractMetric>,
    /// Public serving outputs.
    pub outputs: Vec<ContractPublicOutput>,
}

/// Validated output contract with inferred graph semantics.
#[derive(Debug, Clone)]
pub struct ValidatedOutputContract {
    /// Inferred kind for every graph node.
    pub node_kinds: HashMap<String, ContractNodeKind>,
    /// Stable topological relation order.
    pub relation_order: Vec<String>,
}

impl OutputContract {
    /// Validate structure, infer node kinds and optionally bind raw sources.
    pub fn validate(
        &self,
        sources: Option<&[SourceContract]>,
    ) -> Result<ValidatedOutputContract, String> {
        if self.version != 1 {
            return Err(format!(
                "output_contract.version must be 1, got {}",
                self.version
            ));
        }
        unique_names(
            "graph node",
            self.graph
                .towers
                .iter()
                .map(|v| &v.name)
                .chain(self.graph.relations.iter().map(|v| &v.name)),
        )?;
        unique_names("objective", self.objectives.iter().map(|v| &v.name))?;
        unique_names("metric", self.metrics.iter().map(|v| &v.name))?;
        unique_names("public output", self.outputs.iter().map(|v| &v.name))?;

        let activations = ["relu", "sigmoid", "swish", "gelu", "none"];
        let mut kinds = HashMap::new();
        for tower in &self.graph.towers {
            if tower.kind == ContractNodeKind::Probability {
                return Err(format!(
                    "tower '{}' kind probability is invalid; probability towers are forbidden",
                    tower.name
                ));
            }
            if tower.hidden_dims.contains(&0) {
                return Err(format!(
                    "tower '{}' hidden_dims must contain positive integers",
                    tower.name
                ));
            }
            if !activations.contains(&tower.activation.as_str()) {
                return Err(format!(
                    "tower '{}' has unsupported activation '{}'",
                    tower.name, tower.activation
                ));
            }
            kinds.insert(tower.name.clone(), tower.kind);
        }

        let mut pending: HashMap<&str, &ContractRelation> = self
            .graph
            .relations
            .iter()
            .map(|relation| (relation.name.as_str(), relation))
            .collect();
        let relation_names: HashSet<&str> = pending.keys().copied().collect();
        let mut order = Vec::with_capacity(pending.len());
        while !pending.is_empty() {
            let mut ready: Vec<&str> = pending
                .iter()
                .filter(|(_, relation)| relation.inputs.iter().all(|name| kinds.contains_key(name)))
                .map(|(name, _)| *name)
                .collect();
            ready.sort_unstable();
            if ready.is_empty() {
                let unknown: Vec<&str> = pending
                    .values()
                    .flat_map(|relation| relation.inputs.iter().map(String::as_str))
                    .filter(|name| !kinds.contains_key(*name) && !relation_names.contains(*name))
                    .collect();
                if !unknown.is_empty() {
                    return Err(format!("relation references unknown node(s): {unknown:?}"));
                }
                return Err(format!(
                    "output graph contains a cycle: {:?}",
                    pending.keys().collect::<Vec<_>>()
                ));
            }
            for name in ready {
                let relation = pending.remove(name).expect("ready relation exists");
                let kind = infer_relation_kind(relation, &kinds)?;
                kinds.insert(name.to_string(), kind);
                order.push(name.to_string());
            }
        }

        self.validate_consumers(&kinds)?;
        self.validate_objectives(&kinds)?;
        self.validate_metrics(&kinds)?;
        if let Some(sources) = sources {
            self.validate_sources(sources)?;
        }
        Ok(ValidatedOutputContract {
            node_kinds: kinds,
            relation_order: order,
        })
    }

    /// Serialize normalized semantics as stable canonical JSON.
    pub fn canonical_json(&self) -> Result<String, String> {
        let validated = self.validate(None)?;
        let relation_map: HashMap<&str, &ContractRelation> = self
            .graph
            .relations
            .iter()
            .map(|relation| (relation.name.as_str(), relation))
            .collect();
        let mut towers: Vec<&ContractTower> = self.graph.towers.iter().collect();
        towers.sort_by(|left, right| left.name.cmp(&right.name));
        let relations: Vec<&ContractRelation> = validated
            .relation_order
            .iter()
            .map(|name| relation_map[name.as_str()])
            .collect();
        let mut objectives: Vec<&ContractObjective> = self.objectives.iter().collect();
        objectives.sort_by(|left, right| left.name.cmp(&right.name));
        let objectives: Vec<serde_json::Value> =
            objectives.into_iter().map(canonical_objective).collect();
        let mut metrics: Vec<&ContractMetric> = self.metrics.iter().collect();
        metrics.sort_by(|left, right| left.name.cmp(&right.name));
        let mut outputs: Vec<&ContractPublicOutput> = self.outputs.iter().collect();
        outputs.sort_by(|left, right| left.name.cmp(&right.name));
        serde_json::to_string(&serde_json::json!({
            "graph": {"relations": relations, "towers": towers},
            "metrics": metrics,
            "objectives": objectives,
            "outputs": outputs,
            "version": self.version,
        }))
        .map_err(|error| format!("serialize canonical output contract: {error}"))
    }

    fn validate_consumers(&self, kinds: &HashMap<String, ContractNodeKind>) -> Result<(), String> {
        let mut consumed: HashSet<&str> = self
            .graph
            .relations
            .iter()
            .flat_map(|relation| relation.inputs.iter().map(String::as_str))
            .collect();
        for source in self
            .objectives
            .iter()
            .map(|v| &v.source)
            .chain(self.metrics.iter().map(|v| &v.source))
            .chain(self.outputs.iter().map(|v| &v.source))
        {
            if !kinds.contains_key(source) {
                return Err(format!("'{source}' references unknown output graph node"));
            }
            consumed.insert(source);
        }
        let mut unused: Vec<&str> = kinds
            .keys()
            .map(String::as_str)
            .filter(|name| !consumed.contains(name))
            .collect();
        unused.sort_unstable();
        if !unused.is_empty() {
            return Err(format!("output graph has unused node(s): {unused:?}"));
        }
        Ok(())
    }

    fn validate_objectives(&self, kinds: &HashMap<String, ContractNodeKind>) -> Result<(), String> {
        for objective in &self.objectives {
            if !objective.weight.is_finite() || objective.weight < 0.0 {
                return Err(format!(
                    "objective '{}' weight must be finite and non-negative",
                    objective.name
                ));
            }
            validate_mask(
                objective.mask.as_ref(),
                &format!("objective '{}'", objective.name),
            )?;
            validate_loss(&objective.loss)?;
            let kind = kinds[&objective.source];
            let allowed = match objective.loss.loss_type.as_str() {
                "binary_cross_entropy_with_logits"
                | "focal_binary_cross_entropy_with_logits"
                | "weighted_bce_stay" => kind == ContractNodeKind::BinaryLogit,
                "focal_binary_cross_entropy" | "binary_cross_entropy" => {
                    kind == ContractNodeKind::Probability
                }
                "mse" | "mae" | "huber" => {
                    matches!(kind, ContractNodeKind::Regression | ContractNodeKind::Score)
                }
                _ => false,
            };
            if !allowed {
                return Err(format!(
                    "objective '{}' loss {} cannot consume {:?}",
                    objective.name, objective.loss.loss_type, kind
                ));
            }
        }
        Ok(())
    }

    fn validate_metrics(&self, kinds: &HashMap<String, ContractNodeKind>) -> Result<(), String> {
        for metric in &self.metrics {
            validate_mask(metric.mask.as_ref(), &format!("metric '{}'", metric.name))?;
            let kind = kinds[&metric.source];
            let allowed = match metric.metric_type.as_str() {
                "auc" | "prauc" => matches!(
                    kind,
                    ContractNodeKind::BinaryLogit | ContractNodeKind::Probability
                ),
                "logloss" => kind == ContractNodeKind::Probability,
                "mae" | "mse" => {
                    matches!(kind, ContractNodeKind::Regression | ContractNodeKind::Score)
                }
                _ => {
                    return Err(format!(
                        "metric '{}' type is unsupported",
                        metric.metric_type
                    ))
                }
            };
            if !allowed {
                return Err(format!(
                    "metric '{}' type {} cannot consume {:?}",
                    metric.name, metric.metric_type, kind
                ));
            }
        }
        Ok(())
    }

    fn validate_sources(&self, sources: &[SourceContract]) -> Result<(), String> {
        let catalog: HashMap<&str, &SourceContract> = sources
            .iter()
            .map(|source| (source.name.as_str(), source))
            .collect();
        for source in sources {
            if source.role == "label" && source.has_default {
                return Err(format!(
                    "label source '{}' must not define a default",
                    source.name
                ));
            }
        }
        for (label, mask) in self
            .objectives
            .iter()
            .map(|v| (&v.label, v.mask.as_ref()))
            .chain(self.metrics.iter().map(|v| (&v.label, v.mask.as_ref())))
        {
            let Some(source) = catalog.get(label.as_str()) else {
                return Err(format!("'{label}' must reference a label source"));
            };
            if source.role != "label" {
                return Err(format!("'{label}' must reference a label source"));
            }
            if let Some(mask) = mask {
                if !catalog.contains_key(mask.source.as_str()) {
                    return Err(format!("mask source '{}' does not exist", mask.source));
                }
            }
        }
        Ok(())
    }
}

fn infer_relation_kind(
    relation: &ContractRelation,
    kinds: &HashMap<String, ContractNodeKind>,
) -> Result<ContractNodeKind, String> {
    let input_kinds: Vec<ContractNodeKind> =
        relation.inputs.iter().map(|name| kinds[name]).collect();
    match relation.op {
        ContractRelationOp::Sigmoid => {
            require_arity(relation, 1)?;
            if input_kinds != [ContractNodeKind::BinaryLogit] {
                return Err(format!(
                    "relation '{}' sigmoid requires binary_logit",
                    relation.name
                ));
            }
            Ok(ContractNodeKind::Probability)
        }
        ContractRelationOp::Multiply => {
            if input_kinds.len() < 2
                || input_kinds
                    .iter()
                    .any(|kind| *kind != ContractNodeKind::Probability)
            {
                return Err(format!(
                    "relation '{}' multiply requires at least two probability inputs",
                    relation.name
                ));
            }
            Ok(ContractNodeKind::Probability)
        }
        ContractRelationOp::Add => {
            if input_kinds.len() < 2
                || input_kinds
                    .iter()
                    .any(|kind| *kind != ContractNodeKind::Regression)
            {
                return Err(format!(
                    "relation '{}' add requires at least two regression inputs",
                    relation.name
                ));
            }
            Ok(ContractNodeKind::Regression)
        }
        ContractRelationOp::Identity => {
            require_arity(relation, 1)?;
            Ok(input_kinds[0])
        }
    }
}

fn canonical_objective(objective: &ContractObjective) -> serde_json::Value {
    let epsilon = objective.loss.epsilon.or_else(|| {
        matches!(
            objective.loss.loss_type.as_str(),
            "binary_cross_entropy" | "focal_binary_cross_entropy"
        )
        .then_some(1e-7)
    });
    let delta = objective
        .loss
        .delta
        .or_else(|| (objective.loss.loss_type == "huber").then_some(1.0));
    let gamma = objective.loss.gamma.or_else(|| {
        matches!(
            objective.loss.loss_type.as_str(),
            "focal_binary_cross_entropy" | "focal_binary_cross_entropy_with_logits"
        )
        .then_some(2.0)
    });
    serde_json::json!({
        "label": objective.label,
        "loss": {
            "alpha": objective.loss.alpha.map(canonical_float),
            "delta": delta.map(canonical_float),
            "epsilon": epsilon.map(canonical_float),
            "gamma": gamma.map(canonical_float),
            "pos_weight": objective.loss.pos_weight.map(canonical_float),
            "reduction": objective.loss.reduction,
            "type": objective.loss.loss_type,
        },
        "mask": objective.mask,
        "name": objective.name,
        "source": objective.source,
        "weight": canonical_float(objective.weight),
    })
}

fn canonical_float(value: f64) -> String {
    let formatted = format!("{value:.17e}");
    let (mantissa, exponent) = formatted
        .split_once('e')
        .expect("scientific float formatting contains exponent");
    let exponent = exponent
        .parse::<i32>()
        .expect("scientific float exponent is numeric");
    format!("{mantissa}e{exponent}")
}

fn validate_loss(loss: &ContractLoss) -> Result<(), String> {
    if !["mean", "sum"].contains(&loss.reduction.as_str()) {
        return Err("loss reduction must be mean or sum".into());
    }
    let allowed = match loss.loss_type.as_str() {
        "binary_cross_entropy_with_logits" => (false, true, false, false, false),
        "focal_binary_cross_entropy_with_logits" => (false, false, false, true, true),
        "binary_cross_entropy" => (true, false, false, false, false),
        "focal_binary_cross_entropy" => (true, false, false, true, true),
        "mse" | "mae" | "weighted_bce_stay" => (false, false, false, false, false),
        "huber" => (false, false, true, false, false),
        _ => return Err(format!("loss type '{}' is unsupported", loss.loss_type)),
    };
    if loss.epsilon.is_some() && !allowed.0
        || loss.pos_weight.is_some() && !allowed.1
        || loss.delta.is_some() && !allowed.2
        || loss.alpha.is_some() && !allowed.3
        || loss.gamma.is_some() && !allowed.4
    {
        return Err(format!(
            "loss {} has parameters that are not valid",
            loss.loss_type
        ));
    }
    let epsilon = loss.epsilon.unwrap_or(1e-7);
    if matches!(
        loss.loss_type.as_str(),
        "binary_cross_entropy" | "focal_binary_cross_entropy"
    ) && !(0.0..0.5).contains(&epsilon)
    {
        return Err("binary_cross_entropy epsilon must be between 0 and 0.5".into());
    }
    if loss
        .pos_weight
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("loss pos_weight must be > 0".into());
    }
    if loss
        .delta
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("loss delta must be > 0".into());
    }
    if loss
        .alpha
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err("loss alpha must be between 0 and 1".into());
    }
    let gamma = loss.gamma.unwrap_or(2.0);
    if matches!(
        loss.loss_type.as_str(),
        "focal_binary_cross_entropy" | "focal_binary_cross_entropy_with_logits"
    ) && (!gamma.is_finite() || gamma < 0.0)
    {
        return Err("loss gamma must be >= 0".into());
    }
    Ok(())
}

fn validate_mask(mask: Option<&ContractMask>, context: &str) -> Result<(), String> {
    let Some(mask) = mask else {
        return Ok(());
    };
    let unary = matches!(mask.op.as_str(), "is_null" | "not_null");
    let comparison = matches!(mask.op.as_str(), "eq" | "ne" | "gt" | "ge" | "lt" | "le");
    if !unary && !comparison {
        return Err(format!("{context} mask op '{}' is unsupported", mask.op));
    }
    if unary == mask.value.is_some() {
        return Err(format!(
            "{context} mask value does not match op '{}'",
            mask.op
        ));
    }
    Ok(())
}

fn require_arity(relation: &ContractRelation, count: usize) -> Result<(), String> {
    if relation.inputs.len() != count {
        return Err(format!(
            "relation '{}' {:?} requires {} input(s)",
            relation.name, relation.op, count
        ));
    }
    Ok(())
}

fn unique_names<'a>(context: &str, names: impl Iterator<Item = &'a String>) -> Result<(), String> {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(format!("duplicate {context} name '{name}'"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL: &str = include_str!("../../tests/fixtures/output_contract_canonical.json");

    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        valid: bool,
        error: Option<String>,
        sources: Vec<SourceContract>,
        contract: OutputContract,
    }

    #[test]
    fn shared_output_contract_cases_match_expected_result() {
        let fixtures: Cases = serde_yaml::from_str(include_str!(
            "../../tests/fixtures/output_contract_cases.yaml"
        ))
        .unwrap();
        for case in fixtures.cases {
            let result = case.contract.validate(Some(&case.sources));
            if case.valid {
                let validated = result.unwrap_or_else(|error| {
                    panic!("case '{}' should be valid: {error}", case.name)
                });
                assert_eq!(
                    validated.node_kinds["ctcvr_prob"],
                    ContractNodeKind::Probability
                );
                assert_eq!(
                    validated.relation_order,
                    ["click_prob", "cvr_prob", "ctcvr_prob"]
                );
            } else {
                let error = result.expect_err(&format!("case '{}' should be invalid", case.name));
                assert!(
                    error.contains(case.error.as_deref().unwrap()),
                    "case '{}' error '{error}'",
                    case.name
                );
            }
        }
    }

    #[test]
    fn canonical_json_ignores_declaration_order() {
        let fixtures: Cases = serde_yaml::from_str(include_str!(
            "../../tests/fixtures/output_contract_cases.yaml"
        ))
        .unwrap();
        let original = fixtures
            .cases
            .into_iter()
            .find(|case| case.name == "valid_esmm")
            .unwrap()
            .contract;
        let mut reordered = original.clone();
        reordered.graph.towers.reverse();
        reordered.graph.relations.reverse();
        reordered.objectives.reverse();
        reordered.metrics.reverse();
        reordered.outputs.reverse();

        let expected = CANONICAL.trim();
        assert_eq!(original.canonical_json().unwrap(), expected);
        assert_eq!(reordered.canonical_json().unwrap(), expected);
    }

    #[test]
    fn prauc_accepts_binary_classification_outputs() {
        let fixtures: Cases = serde_yaml::from_str(include_str!(
            "../../tests/fixtures/output_contract_cases.yaml"
        ))
        .unwrap();
        let mut contract = fixtures
            .cases
            .into_iter()
            .find(|case| case.name == "valid_esmm")
            .unwrap()
            .contract;
        contract.metrics[0].metric_type = "prauc".into();

        contract.validate(None).unwrap();
    }
}
