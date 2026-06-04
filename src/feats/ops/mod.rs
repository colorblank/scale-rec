//! 算子库：统一 CustomOp trait 及 7 个内置算子的注册。

pub mod bucketing;
pub mod cross_feature;
pub mod dict_mapper;
pub mod expression;
pub mod feature_hash;
pub mod concat_hash;
pub mod flat_split;
pub mod json_extract_list;
pub mod list_overlap;
pub mod list_string_parser;
pub mod parsed_feature_hash;
pub mod plugin;
pub mod sequence;
pub mod split;
pub mod string_concat;
pub mod string_parser;

pub use bucketing::Bucketing;
pub use cross_feature::CrossFeature;
pub use dict_mapper::DictMapper;
pub use expression::ExpressionOp;
pub use concat_hash::ConcatHash;
pub use feature_hash::FeatureHash;
pub use flat_split::FlatSplit;
pub use json_extract_list::JsonExtractList;
pub use list_overlap::ListOverlap;
pub use list_string_parser::ListStringParser;
pub use plugin::PluginOp;
pub use parsed_feature_hash::ParsedFeatureHash;
pub use sequence::SequenceOp;
pub use split::Split;
pub use string_concat::StringConcat;
pub use string_parser::StringParser;

/// 强类型特征值，替代 `Arc<dyn Any>`，消除 vtable/downcast 开销。
#[derive(Debug, Clone, PartialEq)]
pub enum Fv {
    Int(i32),
    Float(f32),
    Str(String),
    IntList(Vec<i32>),
    FloatList(Vec<f32>),
    StrList(Vec<String>),
}

impl Fv {
    pub fn type_name(&self) -> &str {
        match self {
            Fv::Int(_) => "int",
            Fv::Float(_) => "float",
            Fv::Str(_) => "str",
            Fv::IntList(_) => "list[int]",
            Fv::FloatList(_) => "list[float]",
            Fv::StrList(_) => "list[str]",
        }
    }
}

impl std::fmt::Display for Fv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fv::Int(v) => write!(f, "{}", v),
            Fv::Float(v) => write!(f, "{}", v),
            Fv::Str(v) => write!(f, "{}", v),
            Fv::IntList(v) => write!(f, "{:?}", v),
            Fv::FloatList(v) => write!(f, "{:?}", v),
            Fv::StrList(v) => write!(f, "{:?}", v),
        }
    }
}

pub trait CustomOp: Send + Sync {
    fn name(&self) -> &str;
    fn process(&self, inputs: &[Fv]) -> Result<Fv, String>;

    /// Batch: columnar inputs → Vec of results. Default falls back to row-by-row.
    fn process_batch(&self, inputs: &[&[Fv]], n_rows: usize) -> Result<Vec<Fv>, String> {
        let mut results = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<Fv> = inputs.iter().map(|col| col[i].clone()).collect();
            results.push(self.process(&row)?);
        }
        Ok(results)
    }
}
