//! 算子库：统一 CustomOp trait 及 7 个内置算子的注册。

/// 连续值分桶算子。
pub mod bucketing;
/// 多输入拼接后哈希算子。
pub mod concat_hash;
/// 特征交叉算子。
pub mod cross_feature;
/// 字典映射算子。
pub mod dict_mapper;
/// Rhai 表达式算子。
pub mod expression;
/// DJB2 特征哈希算子。
pub mod feature_hash;
/// 列表打平分割算子。
pub mod flat_split;
/// JSON 数组提取算子。
pub mod json_extract_list;
/// 列表重叠检测算子。
pub mod list_overlap;
/// 字符串列表切分提取算子。
pub mod list_string_parser;
/// 融合预处理哈希算子。
pub mod parsed_feature_hash;
/// 动态加载插件算子。
pub mod plugin;
/// 算子注册中心。
pub mod registry;
/// 序列填充/截断算子。
pub mod sequence;
/// 字符串分割算子。
pub mod split;
/// 字符串拼接算子。
pub mod string_concat;
/// 结构化字符串解析算子。
pub mod string_parser;

/// Bucketing 算子重新导出。
pub use bucketing::Bucketing;
/// ConcatHash 算子重新导出。
pub use concat_hash::ConcatHash;
/// CrossFeature 算子重新导出。
pub use cross_feature::CrossFeature;
/// DictMapper 算子重新导出。
pub use dict_mapper::DictMapper;
/// ExpressionOp 算子重新导出。
pub use expression::ExpressionOp;
/// FeatureHash 算子重新导出。
pub use feature_hash::FeatureHash;
/// FlatSplit 算子重新导出。
pub use flat_split::FlatSplit;
/// JsonExtractList 算子重新导出。
pub use json_extract_list::JsonExtractList;
/// ListOverlap 算子重新导出。
pub use list_overlap::ListOverlap;
/// ListStringParser 算子重新导出。
pub use list_string_parser::ListStringParser;
/// ParsedFeatureHash 算子重新导出。
pub use parsed_feature_hash::ParsedFeatureHash;
/// PluginOp 算子重新导出。
pub use plugin::PluginOp;
/// SequenceOp 算子重新导出。
pub use sequence::SequenceOp;
/// Split 算子重新导出。
pub use split::Split;
/// StringConcat 算子重新导出。
pub use string_concat::StringConcat;
/// StringParser 算子重新导出。
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
    /// 返回特征值类型名称字符串。
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

/// 自定义算子 trait：所有特征算子必须实现。
pub trait CustomOp: Send + Sync {
    /// 返回算子类型名称。
    fn name(&self) -> &str;
    /// 处理单行输入，返回单行输出。
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
