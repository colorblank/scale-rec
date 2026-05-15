//! 算子库：统一 CustomOp trait 及 7 个内置算子的注册。
use std::any::Any;

mod bucketing;
mod cross_feature;
mod dict_mapper;
mod expression;
mod list_overlap;
mod plugin;
mod sequence;
mod string_concat_hash;
mod string_parser;

pub use bucketing::Bucketing;
pub use cross_feature::CrossFeature;
pub use dict_mapper::DictMapper;
pub use expression::ExpressionOp;
pub use list_overlap::ListOverlap;
pub use plugin::PluginOp;
pub use sequence::SequenceOp;
pub use string_concat_hash::StringConcatHash;
pub use string_parser::StringParser;

/// Batch-compatible feature value type.
pub type Fv = std::sync::Arc<dyn Any + Send + Sync>;

pub trait CustomOp: Send + Sync {
    fn name(&self) -> &str;
    fn process(
        &self,
        inputs: &[&(dyn Any + Send + Sync)],
    ) -> Result<Box<dyn Any + Send + Sync>, String>;

    /// Batch process: columnar inputs → Vec of results per row.
    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let mut results: Vec<Fv> = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<&(dyn Any + Send + Sync)> =
                inputs.iter().map(|col| col[i].as_ref()).collect();
            results.push(std::sync::Arc::from(self.process(&row)?));
        }
        Ok(results)
    }
}
