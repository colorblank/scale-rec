//! 特征预处理 Debug 追踪：逐阶段记录每个特征的输入/输出值和异常。
mod tracer;

pub use tracer::{DebugConfig, DebugTracer, SampleTrace, StageTrace, StageType};
