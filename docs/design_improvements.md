# scale-rec 设计分析与改进方案

本文基于当前代码库实现，从架构、功能、性能和工程治理角度梳理现状、风险与改进路线。目标是让后续优化可以按优先级拆分为明确任务，而不是停留在笼统建议。

## 1. 当前架构概览

scale-rec 采用 Python 训练 + Rust 推理的双运行时架构：

- Python 侧使用 PyTorch、pandas/Polars 相关数据处理和 safetensors 导出，主入口集中在 `python/src/train/app/main.py` 与 `python/src/train/training/trainer.py`。
- Rust 侧使用 Candle 承载模型推理，HTTP 服务由 `src/server` 提供，核心推理链路在 `src/server/engine.rs`。
- 双端共享同一份 YAML 特征配置，由 Python `python/src/train/core/config.py` / `python/src/train/core/dag.py` 与 Rust `src/feats/config.rs` / `src/feats/dag.rs` 解析。
- 模型特征规格不在模型配置中重复声明，而是统一来自 `FeatureDag.embeddable_features()`。
- Python 训练后导出 safetensors，Rust 通过 Candle `VarMap` 加载；权重 key 必须与 Rust `VarBuilder::pp()` 路径严格对齐。

这套设计的核心优点是：特征编排有单一来源，训练与推理模型结构可对齐，Rust 线上服务可以避免携带 Python 运行时。同时，当前代码已经具备多模型注册、发布 manifest、Golden consistency、模型 smoke test、算子级测试等基础设施。

## 2. 架构分析

### 2.1 FeatureDag 与特征配置

当前特征系统的边界比较清晰：

- `FlowConfig` 描述 sources、operators、embed 配置。
- `FeatureDag` 完成拓扑排序、算子创建、schema 推导、验证报告和执行。
- Python `FeatureDag.preprocess_batch()` 负责训练侧从 batch rows 到模型输入 tensor。
- Rust `FeatureDag` 同时保留单样本 HashMap 执行路径和批量预编译 plan 路径。

优势：

- YAML 配置是训练和推理共享契约，减少了特征漂移风险。
- Rust 侧已有 `ExecutionPlan`，将算子输入输出预解析为列 id，降低批量推理中的 HashMap 查找开销。
- schema 推导已经开始下沉到 DAG 层，后续可以继续作为运行时契约来源。
- DAG 构建阶段已经有 orphan source/output 检查，能提前发现部分配置问题。

主要风险：

- Rust 单样本执行路径与 plan 批量路径并存，默认值填充、缺失列处理、错误信息可能出现语义差异。
- 默认值解析在多个位置重复存在，例如 `FeatureDag::parse_default()`、`source_default()`、Python `_parse_default()`，长期容易产生训练/推理不一致。
- `execute_batch_precomputed()` 中仍有旧 HashMap 批量路径，与 plan 路径职责重叠，增加维护成本。
- 当前 schema 还没有完全成为唯一执行契约，部分逻辑仍依赖算子本地默认值、字符串判断或 fallback。

改进方向：

- 将 plan 执行路径确立为 Rust 推理主路径，逐步减少或隔离旧批量执行路径。
- 抽出统一默认值解析逻辑，Rust 侧避免 `parse_default()` 与 `source_default()` 两套行为。
- 将 `FeatureSchema` 的 dtype、seq_len、pooling、default、role 作为训练和推理的统一契约。
- 对 DAG validation 增加严格模式，生产加载模型时可以选择 orphan warning 升级为错误。

### 2.2 模型与权重加载

模型层目前采用注册表模式：

- Rust `src/models/mod.rs` 中 `ModelConfig` 通过 `type` 和透传 `params` 构建模型。
- Python `python/src/train/models/__init__.py` 使用注册表构建同名模型。
- GDCN/ESMM、UniMixer、DeepFM、MMoE、LR 等模型已覆盖主要推荐排序场景。
- Rust 加载前会校验 safetensors key 缺失和 shape mismatch，这是非常关键的稳定性防线。

优势：

- 模型配置不重复声明特征，降低配置耦合。
- Rust/Python 模型名称与参数结构相对统一。
- safetensors key/shape 校验能提前暴露权重不兼容问题。
- UniMixer tokenizer 在 Rust 侧作为外部组件注入，避免模型内部重复构建特征规格。

主要风险：

- Rust 与 Python 的模型构建仍是双份实现，新增模型或新层时必须人工维护命名对齐。
- `task_config` 已经支持更灵活的多任务关系，但默认 ESMM/GDCN-ESMM 仍存在 legacy 参数路径，增加理解成本。
- 模型输出任务、label mapping、loss 任务定义分散在 model config、task spec、manifest 和训练逻辑中。

改进方向：

- 为每个模型补充自动化 state_dict key 对齐测试，尤其是新增层或新增任务时。
- 收敛 legacy hidden_dims 参数，长期以 `task_config` / `tasks` 作为多任务配置主入口。
- 在 manifest 中记录模型输出任务、label、relation 和训练指标，并让服务端 `/models` 返回关键元数据。

### 2.3 训练产物与发布

训练侧已有 `TrainingArtifactManager`：

- run 目录保存 checkpoints、best/latest alias、run manifest。
- 发布路径保存 `.safetensors` 和 `.manifest.yaml`。
- manifest 记录 weights、feature config、model config 的 sha256。

这是正确方向，已经解决了“只有权重文件无法恢复训练上下文”的问题。

当前不足：

- `copy_configs` 默认关闭时，manifest 记录的是外部 config 路径及其 sha256，长期归档时依赖外部文件仍然存在。
- Rust `ModelRegistry` 已支持按 serving manifest 的 `feature_config_file`、`model_config_file`、`weights_file` 加载，并支持同一 `model_id` 多版本；但 `/models` 还没有返回 schema hash、tasks、metrics 等完整 serving metadata。
- 默认版本目前按版本字符串取最大值，尚未支持显式 alias、灰度权重或 `versions.yaml` 指针。

改进方向：

- 推荐生产默认开启 `copy_configs`，让发布产物自包含。
- 扩展 `ModelInfo`，返回 schema hash、tasks、metrics 和 loaded_at。
- 增加显式默认版本配置，例如 serving manifest 标记、alias 文件或版本索引文件。

## 3. 功能分析

### 3.1 已具备能力

当前代码库已经具备以下核心功能：

- 共享 YAML 特征编排，覆盖多种 operator。
- Python 单文件训练、discover 训练、多模型训练入口。
- 多任务 loss、评估、early stopping、EMA、checkpoint 和 manifest。
- Rust HTTP 推理，支持 pointwise `/predict` 和 broadcast `/predict/broadcast`。
- Rust 推理侧支持 batch DAG、broadcast user 子图预计算和模型热加载。
- Golden consistency 测试覆盖 Python/Rust 特征处理一致性。
- safetensors 权重 key 与 shape 校验。
- benchmark 工具支持 synthetic 和 discover 输入压测。

这些能力说明项目已经从 demo 形态进入“可工程化迭代”的阶段，下一步重点不是堆功能，而是收紧契约、降低漂移风险和提升吞吐稳定性。

### 3.2 功能缺口

需要优先补齐的功能缺口：

- Typed error：当前 `src/server/routes.rs` 的 `map_predict_error()` 仍依赖字符串包含关系分类错误，未来错误文案变化会导致 HTTP status 不稳定。
- 模型级 schema：registry 对多模型不同 feature config 的支持还不完整。
- 配置兼容策略：feature config、model config、manifest schema 的版本兼容规则尚未系统化。
- 线上可观测性：已有 parse/dag/tensor/forward/response 耗时，但缺少 feature default hit rate、空序列比例、截断次数、batch size、broadcast item count 等指标。
- 数据质量闭环：训练侧有 feature quality summary，但还没有与 manifest、线上日志和服务端统计形成统一链路。

## 4. 性能分析

### 4.1 Rust 推理链路

当前 Rust 推理大致分为：

1. JSON rows 转列式 `FeatureValue`。
2. `ExecutionPlan::execute_plan()` 执行 DAG。
3. 将 embeddable feature 列转 Candle tensor。
4. 模型 forward。
5. tensor 输出转 JSON response。

已有优化：

- DAG 算子执行使用预解析列 id。
- broadcast 模式会先计算 user-only 子图，再对 item batch 复用。
- `InferenceMetrics` 已分段记录主要耗时。
- `feature_column_to_tensor()` 对 sequence pooling 使用配置的 `seq_len` 做 padding/truncation。

性能风险：

- JSON 解析和 `HashMap<String, Value>` 请求结构对高 QPS 场景有明显开销。
- `FeatureValue` 在 DAG 和 tensor 构造之间存在较多 clone。
- `FeatureHash` scalar batch 路径持有全局 write lock 处理整批数据，高并发下可能成为热点。
- `FeatureHash` cache 无大小上限，高基数字段可能造成内存持续增长。
- response 阶段将每个 tensor flatten 后转 `Vec<f32>`，任务数和 batch size 增大时会增加拷贝成本。

改进方向：

- 先用现有 `bench` 固化基线：pointwise/broadcast、不同 batch size、不同模型、不同后端。
- 对 `FeatureHash` 增加缓存开关、最大容量或分片缓存，并暴露 hit/miss/size 指标。
- 对请求结构引入更紧凑的列式 API 可选项，减少 JSON row-wise 解析成本。
- 在 tensor 构造阶段减少中间 clone，优先优化高频 int scalar 和 fixed sequence 两类特征。
- 将 broadcast 的 user/item/cross op 数量、skip op 数量加入调试日志，便于验证预计算是否生效。

### 4.2 Python 训练链路

训练链路主要瓶颈在数据读取和特征预处理：

- `stream_file_batches()` 通过 pandas chunk 读取，再转 `to_dict("records")`。
- `FeatureDag.preprocess_batch()` 对 list[dict] 再转列式，再执行 DAG，再组 tensor。
- `Trainer._train_epoch()` 中每个 batch 同步完成数据读取、预处理、forward、loss、backward。

已有优点：

- batch 级别有 timing log，可以看到 data/preproc/forward/loss/backward 比例。
- `FeatureDag.execute_batch()` 已支持列式输入。
- 评估 batch 会提前收集，避免每次评估重复扫描完整文件。

性能风险：

- pandas chunk 到 dict records 再到 columns 属于重复转换，CPU 和内存开销都较高。
- `build_item_index()` 虽然已用 `itertuples()`，但最终仍构造大字典，物品规模很大时内存压力明显。
- 训练数据 pipeline 没有异步 prefetch，GPU/加速设备训练时可能被 CPU 预处理拖住。
- Python 和 Rust 的 batch DAG 行为仍需持续用 Golden 测试锁住，否则优化过程中容易产生细微差异。

改进方向：

- 让 `stream_file_batches()` 直接 yield `dict[str, list]` 列式 features，避免 `to_dict("records")`。
- `FeatureDag.preprocess_batch()` 优先走列式输入，list[dict] 作为兼容路径。
- 大数据训练引入 Polars LazyFrame 或 Arrow RecordBatch 作为中间格式。
- 增加 prefetch worker，将数据读取和模型训练解耦。
- 对关键算子补充 batch consistency 测试：单样本执行与 batch 执行输出必须一致。

## 5. 改进优先级

### P0：稳定性与契约收敛

1. 修复并保持文档 UTF-8 编码，避免中文文档继续出现乱码。
2. 保持 registry 的多模型独立 schema 模式：按 serving manifest 的 `feature_config_file` 构建 DAG，无 manifest 的旧权重仅作为开发 fallback。
3. 引入 typed inference error，替换 `map_predict_error()` 的字符串匹配。
4. 统一 Rust 默认值解析逻辑，避免 `FeatureDag::parse_default()` 与 `source_default()` 行为分叉。
5. 为 Python/Rust batch DAG 增加更多一致性测试，尤其是 list、flatten、null/default、FeatureHash。

### P1：性能与可观测性

1. 将 Python 训练数据流改为列式 batch，减少 records 往返转换。
2. 为 `FeatureHash` 增加 cache metrics、容量限制和可关闭配置。
3. 扩展 `InferenceMetrics`：
   - batch size
   - item count
   - default hit count/rate
   - empty sequence count
   - truncation count
   - broadcast precompute/remaining DAG 耗时
4. 扩展 `/models` 返回 schema hash、tasks、metrics，并支持显式默认版本/alias 配置。
5. 建立固定压测矩阵，把 benchmark 结果写入 docs 或 CI artifact。

### P2：工程治理

1. 收敛模型多任务配置入口，逐步减少 legacy hidden_dims 参数路径。
2. 为模型 state_dict key 对齐建立自动化测试或导出检查脚本。
3. 让生产发布产物默认自包含 feature config 和 model config。
4. 增加 config schema version 兼容策略，明确哪些字段可选、哪些变更需要重新训练。
5. 将训练侧 feature quality 写入 manifest，并在服务端加载后可查询。

### P3：长期平台化

1. 支持列式请求 API 或二进制协议，服务高吞吐召回/排序场景。
2. 将 operator 注册机制标准化，降低新增算子的 Rust/Python 双端维护成本。
3. 对大规模训练引入 Arrow/Polars-first pipeline 和异步 prefetch。
4. 为线上推理增加 Prometheus/OpenTelemetry 指标导出。
5. 建立模型发布、回滚、灰度和兼容检查流程。

## 6. 推荐执行路线

短期先做 P0，核心目标是“不会加载错模型、不会悄悄特征漂移、错误能稳定分类”。这部分收益最大，也最能降低后续性能优化的返工概率。

中期做 P1，重点优化 Python 列式数据流和 Rust 推理观测。不要在没有基线的情况下先重写高性能路径，应先用 `bench` 和训练 timing log 确认热点。

长期再做 P2/P3，把当前工程化能力提升为稳定平台能力，包括配置版本治理、发布自包含、模型元数据查询和监控体系。

## 7. 验收建议

每一轮改进至少满足以下验收条件：

- Rust：`cargo check`、`cargo test` 通过。
- Python：`PYTHONPATH=python/src uv run pytest python/tests/ -v` 通过。
- 格式：Rust 使用 `cargo fmt`，Python 使用 `uvx ruff format python/src/`。
- 一致性：涉及特征、算子、模型结构或权重命名时，必须跑 Golden consistency 或对应 verify 脚本。
- 性能：涉及推理或训练性能时，必须记录优化前后的 batch size、模型、后端、P50/P95/P99、RPS 或 per-batch timing。

总体建议是先收紧契约，再优化性能，最后平台化。当前代码库已经具备较好的分层基础，后续改进应避免大范围重写，优先在现有 `FeatureDag`、`ModelRegistry`、`Trainer` 和 manifest 体系上增量演进。
