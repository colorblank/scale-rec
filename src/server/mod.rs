//! HTTP 推理服务：多模型管理、热更新、双模式 API。
/// 推理引擎：特征预处理 + 模型前向的编排。
pub mod engine;
/// 模型发布 manifest 解析。
pub mod manifest;
/// 模型注册表：多模型管理 + 热更新。
pub mod registry;
/// Axum HTTP 路由处理。
pub mod routes;
/// 请求耗时追踪。
pub mod tracing;
