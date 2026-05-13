use std::any::Any;

mod bucketing;
mod cross_feature;
mod dict_mapper;
mod expression;
mod plugin;
mod sequence;
mod string_parser;

pub use bucketing::Bucketing;
pub use cross_feature::CrossFeature;
pub use dict_mapper::DictMapper;
pub use expression::ExpressionOp;
pub use plugin::PluginOp;
pub use sequence::SequenceOp;
pub use string_parser::StringParser;

pub trait CustomOp: Send + Sync {
    fn name(&self) -> &str;
    fn process(&self, inputs: &[&(dyn Any + Send + Sync)]) -> Result<Box<dyn Any + Send + Sync>, String>;
}
