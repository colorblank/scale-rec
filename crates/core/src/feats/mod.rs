//! 特征预处理核心引擎：配置、DAG 构建、算子库、执行计划。
pub mod builder;
pub mod config;
pub mod defaults;
pub mod executor;
pub mod feature_info;
pub mod ops;
pub mod schema;
pub mod tensor_utils;

pub use config::FeatureSpec;
