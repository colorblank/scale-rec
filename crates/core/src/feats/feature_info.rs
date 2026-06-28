//! 特征信息视图：从 DAG 构建结果投影，提供模型构建和广播策略所需的元数据查询。
use crate::feats::config::{EmbedConfig, OperatorDef, SourceDef, SourceKind};
use std::collections::HashMap;

/// 特征作用域：原始来源或多个来源组合后的派生作用域。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureScope {
    /// 只依赖用户特征。
    User,
    /// 只依赖物品特征。
    Item,
    /// 只依赖上下文特征。
    Context,
    /// 同时依赖用户和物品特征。
    UserItem,
    /// 同时依赖用户和上下文特征。
    UserContext,
    /// 同时依赖物品和上下文特征。
    ItemContext,
    /// 同时依赖用户、物品和上下文特征。
    UserItemContext,
}

impl FeatureScope {
    /// 从原始 SourceKind 映射到作用域；未声明来源按 item 处理以兼容旧配置。
    pub fn from_source_kind(source: Option<SourceKind>) -> Self {
        match source {
            Some(SourceKind::User) => Self::User,
            Some(SourceKind::Context) => Self::Context,
            Some(SourceKind::Item) | None => Self::Item,
            Some(SourceKind::Label) => {
                unreachable!("label sources do not have a feature scope")
            }
        }
    }

    /// 合并输入作用域，保留 user/item/context 三类来源的组合信息。
    pub fn combine(scopes: &[Self]) -> Self {
        let has_user = scopes.iter().any(|scope| scope.has_user());
        let has_item = scopes.iter().any(|scope| scope.has_item());
        let has_context = scopes.iter().any(|scope| scope.has_context());
        match (has_user, has_item, has_context) {
            (true, true, true) => Self::UserItemContext,
            (true, true, false) => Self::UserItem,
            (true, false, true) => Self::UserContext,
            (false, true, true) => Self::ItemContext,
            (true, false, false) => Self::User,
            (false, true, false) => Self::Item,
            (false, false, true) => Self::Context,
            (false, false, false) => Self::Item,
        }
    }

    /// 是否依赖用户特征。
    pub fn has_user(self) -> bool {
        matches!(
            self,
            Self::User | Self::UserItem | Self::UserContext | Self::UserItemContext
        )
    }

    /// 是否依赖物品特征。
    pub fn has_item(self) -> bool {
        matches!(
            self,
            Self::Item | Self::UserItem | Self::ItemContext | Self::UserItemContext
        )
    }

    /// 是否依赖上下文特征。
    pub fn has_context(self) -> bool {
        matches!(
            self,
            Self::Context | Self::UserContext | Self::ItemContext | Self::UserItemContext
        )
    }
}

/// DAG 的只读元数据视图，用于模型构建和 broadcast 策略计算。
pub struct FeatureInfo {
    sources: HashMap<String, SourceDef>,
    node_defs: HashMap<String, OperatorDef>,
    execution_order: Vec<String>,
}

impl FeatureInfo {
    /// 创建只读特征元数据视图。
    pub fn new(
        sources: HashMap<String, SourceDef>,
        node_defs: HashMap<String, OperatorDef>,
        execution_order: Vec<String>,
    ) -> Self {
        Self {
            sources,
            node_defs,
            execution_order,
        }
    }

    /// 返回所有需要 embedding 的 operator 输出特征。
    pub fn embeddable_features(&self) -> Vec<(&str, &EmbedConfig)> {
        let mut result = Vec::new();
        for (_, op) in &self.node_defs {
            if let Some(ref emb) = op.embed {
                for out_name in &op.outputs {
                    result.push((out_name.as_str(), emb));
                }
            }
        }
        result.sort_by(|a, b| a.0.cmp(b.0));
        result
    }

    /// 返回每个 operator 的来源作用域。
    pub fn op_source_kind(&self) -> HashMap<String, FeatureScope> {
        let mut feat_kind: HashMap<String, FeatureScope> = HashMap::new();
        for (name, src) in &self.sources {
            feat_kind.insert(name.clone(), FeatureScope::from_source_kind(src.source));
        }
        for node_name in &self.execution_order {
            let def = &self.node_defs[node_name];
            let kinds: Vec<FeatureScope> = def
                .inputs
                .iter()
                .filter_map(|inp| feat_kind.get(inp).copied())
                .collect();
            let k = FeatureScope::combine(&kinds);
            for out_name in &def.outputs {
                feat_kind.insert(out_name.clone(), k);
            }
        }
        let mut op_kind: HashMap<String, FeatureScope> = HashMap::new();
        for node_name in &self.execution_order {
            let def = &self.node_defs[node_name];
            let kinds: Vec<FeatureScope> = def
                .inputs
                .iter()
                .filter_map(|inp| feat_kind.get(inp).copied())
                .collect();
            op_kind.insert(node_name.clone(), FeatureScope::combine(&kinds));
        }
        op_kind
    }

    /// 返回 source 定义映射。
    pub fn source_defs(&self) -> &HashMap<String, SourceDef> {
        &self.sources
    }

    /// 返回所有 source 名称。
    pub fn source_names(&self) -> Vec<&str> {
        self.sources.keys().map(|s| s.as_str()).collect()
    }

    /// 返回指定 operator 的输出特征名列表。
    pub fn op_outputs(&self, op_name: &str) -> Option<&Vec<String>> {
        self.node_defs.get(op_name).map(|d| &d.outputs)
    }
}
