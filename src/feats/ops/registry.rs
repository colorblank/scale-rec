//! 算子注册中心：OpFactory 类型别名、全局注册表、创建函数。
use super::CustomOp;
use crate::feats::config::OpType;

/// 算子工厂函数签名：接收 YAML params，返回 `Box<dyn CustomOp>`。
pub type OpFactory = fn(&serde_yaml::Value) -> Result<Box<dyn CustomOp>, String>;

use std::collections::HashMap;
use std::sync::LazyLock;

static OP_REGISTRY: LazyLock<HashMap<OpType, OpFactory>> = LazyLock::new(|| {
    let mut m: HashMap<OpType, OpFactory> = HashMap::new();
    m.insert(OpType::Bucketing, super::bucketing::create);
    m.insert(OpType::ConcatHash, super::concat_hash::create);
    m.insert(OpType::CrossFeature, super::cross_feature::create);
    m.insert(OpType::DictMapper, super::dict_mapper::create);
    m.insert(OpType::ExpressionOp, super::expression::create);
    m.insert(OpType::FeatureHash, super::feature_hash::create);
    m.insert(OpType::FlatSplit, super::flat_split::create);
    m.insert(OpType::JsonExtractList, super::json_extract_list::create);
    m.insert(OpType::ListOverlap, super::list_overlap::create);
    m.insert(OpType::ListStringParser, super::list_string_parser::create);
    m.insert(OpType::Log1p, super::log1p::create);
    m.insert(
        OpType::ParsedFeatureHash,
        super::parsed_feature_hash::create,
    );
    m.insert(OpType::PluginOp, super::plugin::create);
    m.insert(OpType::SequenceOp, super::sequence::create);
    m.insert(OpType::Split, super::split::create);
    m.insert(OpType::StringConcat, super::string_concat::create);
    m.insert(OpType::StringParser, super::string_parser::create);
    m
});

/// 从注册表中查找算子类型并创建实例。
pub fn create_op(op_type: OpType, params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    match OP_REGISTRY.get(&op_type) {
        Some(factory) => factory(params),
        None => Err(format!("Unsupported operator type: {:?}", op_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_17_ops_are_registered() {
        let expected = [
            OpType::Bucketing,
            OpType::ConcatHash,
            OpType::CrossFeature,
            OpType::DictMapper,
            OpType::ExpressionOp,
            OpType::FeatureHash,
            OpType::FlatSplit,
            OpType::JsonExtractList,
            OpType::ListOverlap,
            OpType::ListStringParser,
            OpType::Log1p,
            OpType::ParsedFeatureHash,
            OpType::PluginOp,
            OpType::SequenceOp,
            OpType::Split,
            OpType::StringConcat,
            OpType::StringParser,
        ];
        for op in &expected {
            assert!(OP_REGISTRY.contains_key(op), "{:?} is not registered", op);
        }
        assert_eq!(OP_REGISTRY.len(), expected.len());
    }
}
