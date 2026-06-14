//! 多任务预测塔：TaskTower、MultiTaskTower、任务关系推导。
use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
/// 激活函数类型。
pub enum Activation {
    Relu,
    Sigmoid,
    Swish,
    Gelu,
    #[serde(rename = "none")]
    None_,
}
impl Default for Activation {
    fn default() -> Self {
        Activation::Relu
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// 单任务塔配置：名称、隐藏层维度、输出维度、激活函数。
pub struct TowerConfig {
    pub name: String,
    #[serde(default)]
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    #[serde(default)]
    pub activation: Activation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
/// 任务间概率关系运算类型。
pub enum RelationOp {
    Multiply,
    Add,
    Subtract,
    Divide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// 任务关系定义：目标任务由源任务经运算推导。
pub struct TaskRelation {
    pub target: String,
    pub sources: Vec<String>,
    pub op: RelationOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// 多任务塔完整配置：塔列表 + 关系推导列表。
pub struct MultiTaskConfig {
    pub towers: Vec<TowerConfig>,
    #[serde(default)]
    pub relations: Vec<TaskRelation>,
}

/// 单任务预测塔。
///
/// 多层 MLP + 层间激活，输出 logits（末层无激活）。
/// 参数命名: `hidden.{i}` 和 `output.{num_hidden}` 匹配 Candle 路径。
pub struct TaskTower {
    name: String,
    layers: Vec<Linear>,
    activation: Activation,
}

impl TaskTower {
    /// 根据配置构造塔。`input_dim` 为共享底层输出维度。
    pub fn new(config: &TowerConfig, input_dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut in_dim = input_dim;
        let num_hidden = config.hidden_dims.len();
        for (i, &h_dim) in config.hidden_dims.iter().enumerate() {
            layers.push(linear(in_dim, h_dim, vb.pp(format!("hidden.{}", i)))?);
            in_dim = h_dim;
        }
        layers.push(linear(
            in_dim,
            config.output_dim,
            vb.pp(format!("output.{}", num_hidden)),
        )?);
        Ok(Self {
            name: config.name.clone(),
            layers,
            activation: config.activation.clone(),
        })
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation {
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

    /// 前向: hidden → act → ... → output logits。
    /// 前向：各塔独立输出 → 关系推导（sigmoid 后运算）。
    /// 返回 `{task_name: logits}` 字典。
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            out = layer.forward(&out)?;
            if i < n - 1 {
                out = self.apply_activation(&out)?;
            }
        }
        Ok(out)
    }

    /// 返回该塔的任务名称。
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Module for TaskTower {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

/// 多任务塔管理器。
///
/// 管理独立塔的前向传播和任务间概率关系推导（如 CTCVR = CTR × CVR）。
pub struct MultiTaskTower {
    towers: Vec<TaskTower>,
    relations: Vec<TaskRelation>,
}

impl MultiTaskTower {
    /// 根据配置构建多任务塔。
    pub fn new(config: &MultiTaskConfig, input_dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut towers = Vec::new();
        for tc in &config.towers {
            let tower_vb = vb.pp(tc.name.clone());
            towers.push(TaskTower::new(tc, input_dim, tower_vb)?);
        }
        Ok(Self {
            towers,
            relations: config.relations.clone(),
        })
    }

    /// 返回所有任务名称（含推导任务）。
    pub fn task_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.towers.iter().map(|t| t.name().to_string()).collect();
        for r in &self.relations {
            names.push(r.target.clone());
        }
        names
    }

    /// 返回独立塔的数量（不含推导任务）。
    pub fn num_towers(&self) -> usize {
        self.towers.len()
    }

    /// 前向: hidden → act → ... → output logits。
    /// 前向：各塔独立输出 → 关系推导（sigmoid 后运算）。
    /// 返回 `{task_name: logits}` 字典。
    pub fn forward(&self, shared_output: &Tensor) -> Result<HashMap<String, Tensor>> {
        let mut outputs: HashMap<String, Tensor> = HashMap::new();
        for tower in &self.towers {
            outputs.insert(tower.name().to_string(), tower.forward(shared_output)?);
        }
        for rel in &self.relations {
            let derived = apply_relation(rel, &outputs)?;
            outputs.insert(rel.target.clone(), derived);
        }
        Ok(outputs)
    }

    /// 返回各独立塔的 logits（不含推导任务）。
    pub fn tower_logits(&self, shared_output: &Tensor) -> Result<Vec<Tensor>> {
        self.towers
            .iter()
            .map(|t| t.forward(shared_output))
            .collect()
    }
}

/// 应用任务间概率关系推导（如 CTCVR = sigmoid(click) × sigmoid(cvr)）。
///
/// 从 `outputs` 中取出 `sources` 任务的 logits，经 sigmoid 转换后按 `op` 运算，
/// 返回推导目标的概率值。
pub fn apply_relation(rel: &TaskRelation, outputs: &HashMap<String, Tensor>) -> Result<Tensor> {
    if rel.sources.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "relation '{}' has no sources",
            rel.target
        )));
    }
    let get_prob = |name: &str| -> Result<Tensor> {
        let logit = outputs
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("task '{}' not found", name)))?;
        candle_nn::ops::sigmoid(logit)
    };
    match rel.op {
        RelationOp::Multiply => {
            let mut result = get_prob(&rel.sources[0])?;
            for name in &rel.sources[1..] {
                result = result.mul(&get_prob(name)?)?;
            }
            Ok(result)
        }
        RelationOp::Add => {
            let mut result = get_prob(&rel.sources[0])?;
            for name in &rel.sources[1..] {
                result = result.broadcast_add(&get_prob(name)?)?;
            }
            Ok(result)
        }
        RelationOp::Subtract => {
            if rel.sources.len() != 2 {
                return Err(candle_core::Error::Msg(format!(
                    "relation '{}' subtract requires 2 sources",
                    rel.target
                )));
            }
            get_prob(&rel.sources[0])?.broadcast_sub(&get_prob(&rel.sources[1])?)
        }
        RelationOp::Divide => {
            if rel.sources.len() != 2 {
                return Err(candle_core::Error::Msg(format!(
                    "relation '{}' divide requires 2 sources",
                    rel.target
                )));
            }
            get_prob(&rel.sources[0])?.broadcast_div(&get_prob(&rel.sources[1])?)
        }
    }
}
