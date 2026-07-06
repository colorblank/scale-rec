use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

use scale_rec::feats::config::FlowConfig;
use scale_rec::feats::dag::FeatureDag;
use scale_rec::layers::embedding::FeatureSpec;
use scale_rec::layers::towers::{
    apply_relation, Activation, MultiTaskConfig, RelationOp, TaskRelation, TowerConfig,
};
use scale_rec::models::{
    deepfm, esmm, gdcn_esmm, lr, mmoe, pepnet, Model, ModelConfig, ModelOutput, OutputKind,
};

fn dummy_features() -> Vec<FeatureSpec> {
    vec![
        FeatureSpec::new("a".into(), 10, 4),
        FeatureSpec::new("b".into(), 5, 4),
    ]
}

fn dummy_inputs(batch: usize) -> HashMap<String, Tensor> {
    let mut m = HashMap::new();
    m.insert(
        "a".into(),
        Tensor::from_slice(&[1u32, 2, 3][..batch], batch, &Device::Cpu).unwrap(),
    );
    m.insert(
        "b".into(),
        Tensor::from_slice(&[0u32, 1, 2][..batch], batch, &Device::Cpu).unwrap(),
    );
    m
}

fn demo_features() -> Vec<FeatureSpec> {
    let path = format!(
        "{}/examples/shared/feature_config_demo.yaml",
        env!("CARGO_MANIFEST_DIR")
    );
    let yaml = std::fs::read_to_string(path).unwrap();
    let config = FlowConfig::from_yaml(&yaml).unwrap();
    let dag = FeatureDag::from_config(config, false, None).unwrap();
    dag.embeddable_features()
        .into_iter()
        .map(|(name, emb)| FeatureSpec {
            name: name.to_string(),
            vocab_size: emb.vocab_size,
            embed_dim: emb.embed_dim,
            pooling: emb.pooling,
            seq_len: emb.seq_len,
            truncation: emb.truncation,
        })
        .collect()
}

fn zero_inputs(features: &[FeatureSpec], batch: usize) -> HashMap<String, Tensor> {
    features
        .iter()
        .map(|feature| {
            let shape = if let Some(seq_len) = feature.seq_len {
                vec![batch, seq_len]
            } else {
                vec![batch]
            };
            let len = shape.iter().product();
            let tensor = Tensor::from_vec(vec![0u32; len], shape, &Device::Cpu).unwrap();
            (feature.name.clone(), tensor)
        })
        .collect()
}

fn vb() -> VarBuilder<'static> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    // leak to get 'static
    Box::leak(Box::new(varmap));
    vb
}

#[test]
fn test_lr_forward_shape() {
    let model = lr::LogisticRegression::new(vb(), &dummy_features()).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    let pred = out.get("pred").unwrap();
    assert_eq!(pred.tensor.dims(), &[3, 1]);
}

#[test]
fn test_deepfm_forward_shape() {
    let model = deepfm::DeepFM::new(vb(), &dummy_features(), 8, &[4]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.tensor("pred").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_mmoe_forward_shape() {
    let task_cfgs = vec![("ctr".into(), vec![4]), ("cvr".into(), vec![4])];
    let model = mmoe::MMoE::new(vb(), &dummy_features(), &[8], 2, &[8], 4, &task_cfgs).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out.tensor("ctr").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("cvr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_esmm_forward_shape() {
    let model =
        esmm::ESMM::new(vb(), &dummy_features(), &[8], &[4], &[4], &[4], &[4], &[4]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 9);
    assert_eq!(out.tensor("click").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("cvr").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("ctcvr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_esmm_forward_with_configurable_tasks_and_relations() {
    let task_config = MultiTaskConfig {
        towers: vec![
            TowerConfig {
                name: "view".into(),
                hidden_dims: vec![4],
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
            TowerConfig {
                name: "buy".into(),
                hidden_dims: vec![4],
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
        ],
        relations: vec![TaskRelation {
            target: "ctbuy".into(),
            sources: vec!["view".into(), "buy".into()],
            op: RelationOp::Multiply,
        }],
    };
    let model = esmm::ESMM::with_task_config(vb(), &dummy_features(), &[8], &task_config).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out.tensor("view").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("buy").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("ctbuy").unwrap().dims(), &[3, 1]);
    assert_eq!(out.get("view").unwrap().kind, OutputKind::BinaryLogit);
    assert_eq!(out.get("ctbuy").unwrap().kind, OutputKind::Probability);
}

#[test]
fn test_task_relation_uses_probabilities_not_logits() {
    let relation = TaskRelation {
        target: "ctbuy".into(),
        sources: vec!["view".into(), "buy".into()],
        op: RelationOp::Multiply,
    };
    let mut outputs = ModelOutput::new();
    outputs.insert_binary_logit(
        "view",
        Tensor::from_slice(&[0.0f32, 2.0], (2, 1), &Device::Cpu).unwrap(),
    );
    outputs.insert_binary_logit(
        "buy",
        Tensor::from_slice(&[0.0f32, -2.0], (2, 1), &Device::Cpu).unwrap(),
    );

    let derived = apply_relation(&relation, &outputs).unwrap();
    let values = derived.to_vec2::<f32>().unwrap();
    let expected0 = 0.5 * 0.5;
    let expected1 = sigmoid(2.0) * sigmoid(-2.0);

    assert!((values[0][0] - expected0).abs() < 1e-6);
    assert!((values[1][0] - expected1).abs() < 1e-6);
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn test_gdcn_esmm_forward_shape() {
    let model = gdcn_esmm::GDCNESMM::new(
        vb(),
        &dummy_features(),
        2,
        &[8],
        &[8],
        &[4],
        &[4],
        &[4],
        &[4],
        &[4],
    )
    .unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 9);
    assert_eq!(out.tensor("click").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("cvr").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("ctcvr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_pepnet_forward_shape() {
    let tc = MultiTaskConfig {
        towers: vec![
            TowerConfig {
                name: "click".into(),
                hidden_dims: vec![4],
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
            TowerConfig {
                name: "cvr".into(),
                hidden_dims: vec![4],
                output_dim: 1,
                activation: Activation::Relu,
                output_kind: OutputKind::BinaryLogit,
            },
        ],
        relations: vec![TaskRelation {
            target: "ctcvr".into(),
            sources: vec!["click".into(), "cvr".into()],
            op: RelationOp::Multiply,
        }],
    };
    let model = pepnet::PEPNet::new(vb(), &dummy_features(), 4, &[8], &[8], &tc, &[], &[]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out.tensor("click").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("cvr").unwrap().dims(), &[3, 1]);
    assert_eq!(out.tensor("ctcvr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_pepnet_forward_shape_with_deep_without_shared_bottom() {
    let tc = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "click".into(),
            hidden_dims: vec![4],
            output_dim: 1,
            activation: Activation::Relu,
            output_kind: OutputKind::BinaryLogit,
        }],
        relations: vec![],
    };
    let model = pepnet::PEPNet::new(vb(), &dummy_features(), 4, &[8], &[], &tc, &[], &[]).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out.tensor("click").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_modelconfig_build_gdcn_esmm() {
    let params = serde_yaml::from_str(
        r#"
cross_layers: 2
deep_hidden_dims: [8]
shared_bottom_dims: [8]
task_config:
  towers:
    - {name: view, hidden_dims: [4], output_dim: 1, activation: relu}
    - {name: buy, hidden_dims: [4], output_dim: 1, activation: relu}
  relations:
    - {target: ctbuy, sources: [view, buy], op: multiply}
"#,
    )
    .unwrap();
    let cfg = ModelConfig {
        model_type: "gdcn_esmm".into(),
        params,
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out.tensor("view").unwrap().dims(), &[2, 1]);
    assert_eq!(out.tensor("buy").unwrap().dims(), &[2, 1]);
    assert_eq!(out.tensor("ctbuy").unwrap().dims(), &[2, 1]);
}

#[test]
fn test_modelconfig_build_native_output_contract_esmm() {
    let config: ModelConfig =
        serde_yaml::from_str(include_str!("../examples/models/esmm_output_contract.yaml")).unwrap();
    let model = config.build(vb(), &dummy_features(), None).unwrap();

    let public = model.forward(&dummy_inputs(2)).unwrap();
    let execution = model.forward_execution(&dummy_inputs(2)).unwrap();

    assert_eq!(public.len(), 5);
    assert!(public.contains_key("ctr"));
    assert!(public.contains_key("ctcvr"));
    assert!(!public.contains_key("click_logit"));
    assert!(execution.nodes.contains_key("click_logit"));
    assert!(execution.nodes.contains_key("ctcvr_prob"));
    assert_eq!(
        execution.outputs.get("ctcvr").unwrap().kind,
        OutputKind::Probability
    );
}

#[test]
fn test_all_example_models_build_with_native_output_contract() {
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let configs = [
        "lr.yaml",
        "deepfm.yaml",
        "mmoe.yaml",
        "esmm_output_contract.yaml",
        "gdcn_esmm.yaml",
        "unimixer.yaml",
        "token_mixer_large.yaml",
        "rankmixer.yaml",
        "full_mix.yaml",
        "rankup.yaml",
        "hyformer.yaml",
        "onetrans.yaml",
        "uniformer.yaml",
        "pepnet.yaml",
    ];
    for name in configs {
        let path = format!("{}/examples/models/{name}", env!("CARGO_MANIFEST_DIR"));
        let yaml = std::fs::read_to_string(path).unwrap();
        let config: ModelConfig = serde_yaml::from_str(&yaml).unwrap();
        let builder = vb();
        let features = if config.model_type == "pepnet" {
            demo_features()
        } else {
            dummy_features()
        };
        let inputs = if config.model_type == "pepnet" {
            zero_inputs(&features, 2)
        } else {
            dummy_inputs(2)
        };
        let tokenizer = if matches!(
            config.model_type.as_str(),
            "unimixer" | "token_mixer_large" | "rankmixer" | "full_mix"
        ) {
            let token_dim = config.params["token_dim"].as_u64().unwrap() as usize;
            let num_tokens = config.params["num_tokens"].as_u64().unwrap() as usize;
            Some(
                FeatureTokenizer::new(builder.pp("tokenizer"), &features, token_dim, num_tokens)
                    .unwrap(),
            )
        } else {
            None
        };

        let model = config
            .build(builder, &features, tokenizer)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let execution = model
            .forward_execution(&inputs)
            .unwrap_or_else(|error| panic!("{name}: {error}"));

        assert!(!execution.nodes.is_empty(), "{name}");
        assert!(!execution.outputs.is_empty(), "{name}");
    }
}

#[test]
fn test_modelconfig_build_lr() {
    let cfg = ModelConfig {
        model_type: "lr".into(),
        params: serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("pred"));
}

#[test]
fn test_modelconfig_build_deepfm() {
    let mut params = serde_yaml::Mapping::new();
    params.insert("fm_k".into(), serde_yaml::Value::Number(8.into()));
    let mut dims = serde_yaml::Sequence::new();
    dims.push(serde_yaml::Value::Number(4.into()));
    params.insert("deep_hidden_dims".into(), serde_yaml::Value::Sequence(dims));
    let cfg = ModelConfig {
        model_type: "deepfm".into(),
        params: serde_yaml::Value::Mapping(params),
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("pred"));
}

#[test]
fn test_unimixer_forward_shape() {
    use scale_rec::models::unimixer::{model::UniMixerModel, tokenizer::FeatureTokenizer};

    let features = dummy_features();
    let token_dim = 4;
    let num_tokens = 2;
    let vb = vb();
    let tokenizer =
        FeatureTokenizer::new(vb.pp("tokenizer"), &features, token_dim, num_tokens).unwrap();
    let task_config = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "ctr".into(),
            hidden_dims: vec![8],
            output_dim: 1,
            activation: Activation::Relu,
            output_kind: OutputKind::BinaryLogit,
        }],
        relations: vec![],
    };
    let model = UniMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        1,
        Some(4),
        false,
        1.0,
        4,
        4,
        &task_config,
        false,
        vb.pp("unimixer"),
    )
    .unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert!(out.contains_key("ctr"));
    assert_eq!(out.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_rankmixer_forward_shape() {
    use scale_rec::models::rankmixer::model::RankMixerModel;
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let features = dummy_features();
    let token_dim = 4;
    let num_tokens = 2;
    let vb = vb();
    let tokenizer =
        FeatureTokenizer::new(vb.pp("tokenizer"), &features, token_dim, num_tokens).unwrap();
    let task_config = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "ctr".into(),
            hidden_dims: vec![8],
            output_dim: 1,
            activation: Activation::Relu,
            output_kind: OutputKind::BinaryLogit,
        }],
        relations: vec![],
    };
    let model = RankMixerModel::new(
        tokenizer,
        token_dim,
        num_tokens,
        1,
        num_tokens,
        1.0,
        &task_config,
        vb,
    )
    .unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert!(out.contains_key("ctr"));
    assert_eq!(out.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_full_mix_forward_shape() {
    use scale_rec::models::full_mix::model::FullMixModel;
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let features = dummy_features();
    let token_dim = 4;
    let num_tokens = 2;
    let vb = vb();
    let tokenizer =
        FeatureTokenizer::new(vb.pp("tokenizer"), &features, token_dim, num_tokens).unwrap();
    let task_config = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "ctr".into(),
            hidden_dims: vec![8],
            output_dim: 1,
            activation: Activation::Relu,
            output_kind: OutputKind::BinaryLogit,
        }],
        relations: vec![],
    };
    let model =
        FullMixModel::new(tokenizer, token_dim, num_tokens, 1, 2.0, &task_config, vb).unwrap();
    let out = model.forward(&dummy_inputs(3)).unwrap();
    assert!(out.contains_key("ctr"));
    assert_eq!(out.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_rankup_forward_with_task_token_output_contract() {
    use scale_rec::models::rankup::model::{RankUpConfig, RankUpModel};

    let features = dummy_features();
    let contract: scale_rec::models::output_contract::OutputContract = serde_yaml::from_str(
        r#"
version: 1
graph:
  towers:
    - {name: ctr_logit, input: task_0, kind: binary_logit, hidden_dims: [4]}
  relations:
    - {name: ctr_prob, op: sigmoid, inputs: [ctr_logit]}
objectives:
  - {name: ctr_loss, source: ctr_logit, label: is_click, loss: {type: binary_cross_entropy_with_logits}}
metrics:
  - {name: ctr_auc, source: ctr_logit, label: is_click, type: auc}
outputs:
  - {name: ctr, source: ctr_prob}
"#,
    )
    .unwrap();
    let model = RankUpModel::with_output_contract(
        vb(),
        &features,
        RankUpConfig {
            token_dim: 4,
            num_sparse_tokens: 2,
            num_blocks: 1,
            num_heads: None,
            hidden_factor: 1.0,
            permutation_seed: 2026,
            multi_embedding_tables: 1,
            use_global_token: true,
            cross_token: None,
            num_task_tokens: 1,
        },
        &contract,
    )
    .unwrap();

    let execution = model.forward_execution(&dummy_inputs(3)).unwrap();

    assert_eq!(execution.nodes.tensor("ctr_logit").unwrap().dims(), &[3, 1]);
    assert_eq!(execution.outputs.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_hyformer_forward_with_output_contract() {
    use scale_rec::models::hyformer::model::{HyFormerConfig, HyFormerModel};

    let features = dummy_features();
    let contract: scale_rec::models::output_contract::OutputContract = serde_yaml::from_str(
        r#"
version: 1
graph:
  towers:
    - {name: ctr_logit, kind: binary_logit, hidden_dims: [4]}
  relations:
    - {name: ctr_prob, op: sigmoid, inputs: [ctr_logit]}
objectives:
  - {name: ctr_loss, source: ctr_logit, label: is_click, loss: {type: binary_cross_entropy_with_logits}}
metrics:
  - {name: ctr_auc, source: ctr_logit, label: is_click, type: auc}
outputs:
  - {name: ctr, source: ctr_prob}
"#,
    )
    .unwrap();
    let model = HyFormerModel::with_output_contract(
        vb(),
        &features,
        HyFormerConfig {
            d: 4,
            d_ff: 8,
            num_queries: 2,
            num_layers: 1,
            hidden_factor: 1.0,
        },
        &contract,
    )
    .unwrap();

    let execution = model.forward_execution(&dummy_inputs(3)).unwrap();

    assert_eq!(execution.nodes.tensor("ctr_logit").unwrap().dims(), &[3, 1]);
    assert_eq!(execution.outputs.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_onetrans_forward_with_output_contract() {
    use scale_rec::models::onetrans::model::{OneTransConfig, OneTransModel};

    let features = dummy_features();
    let contract: scale_rec::models::output_contract::OutputContract = serde_yaml::from_str(
        r#"
version: 1
graph:
  towers:
    - {name: ctr_logit, kind: binary_logit, hidden_dims: [4]}
  relations:
    - {name: ctr_prob, op: sigmoid, inputs: [ctr_logit]}
objectives:
  - {name: ctr_loss, source: ctr_logit, label: is_click, loss: {type: binary_cross_entropy_with_logits}}
metrics:
  - {name: ctr_auc, source: ctr_logit, label: is_click, type: auc}
outputs:
  - {name: ctr, source: ctr_prob}
"#,
    )
    .unwrap();
    let model = OneTransModel::with_output_contract(
        vb(),
        &features,
        OneTransConfig {
            d: 4,
            d_ff: 8,
            num_layers: 1,
            n_heads: 2,
            pyramid_tail_tokens: None,
        },
        &contract,
    )
    .unwrap();

    let execution = model.forward_execution(&dummy_inputs(3)).unwrap();

    assert_eq!(execution.nodes.tensor("ctr_logit").unwrap().dims(), &[3, 1]);
    assert_eq!(execution.outputs.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_uniformer_forward_with_output_contract() {
    use scale_rec::models::uniformer::model::{UniFormerConfig, UniFormerModel};

    let features = dummy_features();
    let contract: scale_rec::models::output_contract::OutputContract = serde_yaml::from_str(
        r#"
version: 1
graph:
  towers:
    - {name: ctr_logit, input: task_0, kind: binary_logit, hidden_dims: [4]}
  relations:
    - {name: ctr_prob, op: sigmoid, inputs: [ctr_logit]}
objectives:
  - {name: ctr_loss, source: ctr_logit, label: is_click, loss: {type: binary_cross_entropy_with_logits}}
metrics:
  - {name: ctr_auc, source: ctr_logit, label: is_click, type: auc}
outputs:
  - {name: ctr, source: ctr_prob}
"#,
    )
    .unwrap();
    let model = UniFormerModel::with_output_contract(
        vb(),
        &features,
        UniFormerConfig {
            d: 4,
            d_ff: 8,
            num_layers: 1,
            n_heads: 2,
            num_tasks: 1,
        },
        &contract,
    )
    .unwrap();

    let execution = model.forward_execution(&dummy_inputs(3)).unwrap();

    assert_eq!(execution.nodes.tensor("ctr_logit").unwrap().dims(), &[3, 1]);
    assert_eq!(execution.outputs.tensor("ctr").unwrap().dims(), &[3, 1]);
}

#[test]
fn test_modelconfig_build_rankmixer() {
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let features = dummy_features();
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &features, 4, 2).unwrap();
    let params = serde_yaml::from_str(
        r#"
token_dim: 4
num_tokens: 2
num_blocks: 1
task_config:
  towers:
    - {name: ctr, hidden_dims: [8], output_dim: 1, activation: relu}
"#,
    )
    .unwrap();
    let cfg = ModelConfig {
        model_type: "rankmixer".into(),
        params,
    };
    let model = cfg.build(vb, &features, Some(tokenizer)).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert_eq!(out.tensor("ctr").unwrap().dims(), &[2, 1]);
}

#[test]
fn test_rankmixer_rejects_non_residual_token_mixing_shape() {
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let features = dummy_features();
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &features, 4, 2).unwrap();
    let params = serde_yaml::from_str(
        r#"
token_dim: 4
num_tokens: 2
num_heads: 1
task_config:
  towers:
    - {name: ctr, hidden_dims: [8], output_dim: 1, activation: relu}
"#,
    )
    .unwrap();
    let cfg = ModelConfig {
        model_type: "rankmixer".into(),
        params,
    };

    let err = match cfg.build(vb, &features, Some(tokenizer)) {
        Ok(_) => panic!("invalid RankMixer shape should fail"),
        Err(err) => err,
    };
    assert!(err
        .to_string()
        .contains("RankMixer requires num_heads == num_tokens"));
}

#[test]
fn test_modelconfig_unimixer_rejects_invalid_task_config() {
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let features = dummy_features();
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &features, 4, 2).unwrap();
    let params = serde_yaml::from_str(
        r#"
token_dim: 4
num_tokens: 2
task_config: 1
"#,
    )
    .unwrap();
    let cfg = ModelConfig {
        model_type: "unimixer".into(),
        params,
    };

    let err = match cfg.build(vb, &features, Some(tokenizer)) {
        Ok(_) => panic!("invalid task_config should fail"),
        Err(err) => err,
    };
    assert!(err
        .to_string()
        .contains("params.task_config must be mapping"));
}

#[test]
fn test_modelconfig_rejects_wrong_param_type_before_defaulting() {
    let features = dummy_features();
    let vb = vb();
    let params = serde_yaml::from_str(
        r#"
cross_layers: "3"
"#,
    )
    .unwrap();
    let cfg = ModelConfig {
        model_type: "gdcn_esmm".into(),
        params,
    };

    let err = match cfg.build(vb, &features, None) {
        Ok(_) => panic!("wrong param type should fail"),
        Err(err) => err,
    };
    assert!(err
        .to_string()
        .contains("params.cross_layers must be non-negative integer"));
}

#[test]
fn test_unimixer_tokenizer_accepts_sequence_pooling() {
    use scale_rec::feats::config::PoolingStrategy;
    use scale_rec::models::unimixer::tokenizer::FeatureTokenizer;

    let device = Device::Cpu;
    let mut seq = FeatureSpec::new("seq".into(), 16, 4);
    seq.pooling = PoolingStrategy::First;
    let mut flat = FeatureSpec::new("flat".into(), 16, 2);
    flat.pooling = PoolingStrategy::Flatten;
    flat.seq_len = Some(2);
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &[seq, flat], 3, 2).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(
        "seq".into(),
        Tensor::from_slice(&[1u32, 2, 3, 4], (2, 2), &device).unwrap(),
    );
    inputs.insert(
        "flat".into(),
        Tensor::from_slice(&[1u32, 2, 3, 4], (2, 2), &device).unwrap(),
    );

    let out = tokenizer.forward(&inputs).unwrap();

    assert_eq!(out.dims(), &[2, 2, 3]);
}

#[test]
fn test_unimixer_rejects_invalid_temperature() {
    use scale_rec::models::unimixer::{model::UniMixerModel, tokenizer::FeatureTokenizer};

    let features = dummy_features();
    let vb = vb();
    let tokenizer = FeatureTokenizer::new(vb.pp("tokenizer"), &features, 4, 2).unwrap();
    let task_config = MultiTaskConfig {
        towers: vec![TowerConfig {
            name: "ctr".into(),
            hidden_dims: vec![8],
            output_dim: 1,
            activation: Activation::Relu,
            output_kind: OutputKind::BinaryLogit,
        }],
        relations: vec![],
    };
    let model = UniMixerModel::new(
        tokenizer,
        4,
        2,
        1,
        Some(4),
        false,
        1.0,
        4,
        4,
        &task_config,
        false,
        vb.pp("unimixer"),
    )
    .unwrap();

    let err = model
        .forward_with_temperature(&dummy_inputs(3), 0.0)
        .unwrap_err();
    assert!(err.to_string().contains("temperature must be > 0"));
}

#[test]
fn test_modelconfig_build_mmoe() {
    let mut params = serde_yaml::Mapping::new();
    params.insert("num_experts".into(), serde_yaml::Value::Number(2.into()));
    let mut expert_dims = serde_yaml::Sequence::new();
    expert_dims.push(serde_yaml::Value::Number(8.into()));
    params.insert(
        "expert_hidden_dims".into(),
        serde_yaml::Value::Sequence(expert_dims),
    );
    params.insert(
        "expert_output_dim".into(),
        serde_yaml::Value::Number(4.into()),
    );
    let mut tcs = serde_yaml::Sequence::new();
    let mut entry = serde_yaml::Mapping::new();
    entry.insert("name".into(), serde_yaml::Value::String("ctr".into()));
    let mut td = serde_yaml::Sequence::new();
    td.push(serde_yaml::Value::Number(4.into()));
    entry.insert("tower_dims".into(), serde_yaml::Value::Sequence(td));
    tcs.push(serde_yaml::Value::Mapping(entry));
    params.insert("task_configs".into(), serde_yaml::Value::Sequence(tcs));
    let cfg = ModelConfig {
        model_type: "mmoe".into(),
        params: serde_yaml::Value::Mapping(params),
    };
    let model = cfg.build(vb(), &dummy_features(), None).unwrap();
    let out = model.forward(&dummy_inputs(2)).unwrap();
    assert!(out.contains_key("ctr"));
}
