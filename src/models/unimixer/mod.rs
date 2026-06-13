//! UniMixer：Token 化 + 双随机矩阵交互 + 孪生归一化。
/// UniMixerModel 完整模型实现。
pub mod model;
/// PerTokenSwiGLU 激活与投影层。
pub mod per_token_swiglu;
/// 性能分析工具：按阶段计时。
pub mod profile;
/// SiameseNorm 双流归一化。
pub mod siamese_norm;
/// FeatureTokenizer 特征分词器。
pub mod tokenizer;
/// UniMixerBlock 单层交互块。
pub mod unimixer_block;
/// 标准版双随机矩阵交互。
pub mod unimixing;
/// 轻量版基组合 + 低秩近似交互。
pub mod unimixing_lite;
