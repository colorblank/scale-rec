//! 可复用神经网络层：Embedding、MLP、FM 交互、多任务塔。
/// 特征嵌入层：多特征 Embedding 查找与池化。
pub mod embedding;
/// FM 二阶交互层。
pub mod fm;
/// 门控深度交叉网络层。
pub mod gdcn;
/// 通用多层感知机层。
pub mod mlp;
/// 多任务预测塔层。
pub mod towers;
