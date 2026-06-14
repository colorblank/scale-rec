//! 特征预处理引擎：配置、DAG 构建、执行、算子库、监控指标。
/// 特征配置：FlowConfig、SourceDef、OperatorDef 等 YAML 反序列化类型。
pub mod config;
/// 特征 DAG 执行引擎：拓扑排序、算子调度、样本执行。
pub mod dag;
/// 调试追踪：逐阶段记录特征处理的输入/输出。
pub mod debug;
/// 默认值解析：根据 SourceDef dtype 生成 Fv 默认值。
pub mod defaults;
/// 监控指标：数据漂移检测、缺失率统计、性能埋点。
pub mod metrics;
/// 算子库：CustomOp trait 及全部内置算子实现。
pub mod ops;
/// 特征 schema 推断与验证。
pub mod schema;
/// DAG 构建器：FlowConfig → 拓扑排序 → 校验 → 预编译 ExecutionPlan。
pub mod builder;
/// DAG 执行器：ExecutionPlan + DagExecutor，统一 plan-based 执行路径。
pub mod executor;
/// 特征信息视图：embeddable features、op_source_kind 等元数据查询。
pub mod feature_info;
