//! 算子注册中心：OpFactory 类型别名、全局注册表、创建函数。
use super::CustomOp;

/// 算子工厂函数签名：接收 YAML params，返回 Box<dyn CustomOp>
pub type OpFactory = fn(&serde_yaml::Value) -> Result<Box<dyn CustomOp>, String>;

use std::collections::HashMap;
use std::sync::LazyLock;

static OP_REGISTRY: LazyLock<HashMap<&'static str, OpFactory>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, OpFactory> = HashMap::new();
    m.insert("Bucketing", super::bucketing::create);
    m.insert("ConcatHash", super::concat_hash::create);
    m.insert("CrossFeature", super::cross_feature::create);
    m.insert("DictMapper", super::dict_mapper::create);
    m.insert("ExpressionOp", super::expression::create);
    m.insert("FeatureHash", super::feature_hash::create);
    m.insert("FlatSplit", super::flat_split::create);
    m.insert("JsonExtractList", super::json_extract_list::create);
    m.insert("ListOverlap", super::list_overlap::create);
    m.insert("ListStringParser", super::list_string_parser::create);
    m.insert("ParsedFeatureHash", super::parsed_feature_hash::create);
    m.insert("PluginOp", super::plugin::create);
    m.insert("SequenceOp", super::sequence::create);
    m.insert("Split", super::split::create);
    m.insert("StringConcat", super::string_concat::create);
    m.insert("StringParser", super::string_parser::create);
    m
});

/// 从注册表中查找算子类型并创建实例。
pub fn create_op(op_type: &str, params: &serde_yaml::Value) -> Result<Box<dyn CustomOp>, String> {
    match OP_REGISTRY.get(op_type) {
        Some(factory) => factory(params),
        None => Err(format!("Unsupported operator type: {}", op_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_16_ops_are_registered() {
        let expected = [
            "Bucketing", "ConcatHash", "CrossFeature", "DictMapper",
            "ExpressionOp", "FeatureHash", "FlatSplit", "JsonExtractList",
            "ListOverlap", "ListStringParser", "ParsedFeatureHash", "PluginOp",
            "SequenceOp", "Split", "StringConcat", "StringParser",
        ];
        for op in &expected {
            assert!(OP_REGISTRY.contains_key(op), "{} is not registered", op);
        }
        assert_eq!(OP_REGISTRY.len(), expected.len());
    }
}
