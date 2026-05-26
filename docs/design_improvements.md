# 设计改进建议

本文只记录当前仍需要推进的设计和工程改进。已经落地的内容不再放在主 backlog 中，例如 Python/Rust `FeatureSchema` 推导、Python `TaskSpec`、模型 manifest 导出与 Rust manifest 校验、训练包目录重构、discover 配置生成抽象、基础 feature quality 统计、UniMixer 拆分和类型注释补齐等。

## P0：线上推理链路正确性

### 1. 修正 `ModelRegistry::load_model()` 的 feature cache 风险

`src/server/registry.rs` 在 `ModelRegistry::new()` 初始化 `embed_features_cache`，但 `load_model()` 重新解析 feature config 后，构建 tokenizer/model 时仍使用缓存中的 `cached_features`。如果热加载期间 feature config 变化，engine 使用的新 DAG 特征和模型构建使用的特征可能不一致。

建议删除跨加载缓存，或在 `load_model()` 内使用当前解析出的 `embed_features` 原子构建 DAG、tokenizer 和 model。加载完成前应校验 DAG embeddable features、model input features、safetensors key manifest 三者一致。

### 2. 收敛 Rust DAG 执行路径和 broadcast 复用边界

Rust 侧同时存在 single-row、batch、plan、broadcast 等执行路径。路径越多，越容易出现语义漂移。`predict_broadcast()` 当前还依赖 user-only plan 判断，并逐 item clone user row，正确性边界和性能成本都需要更明确。

建议以编译后的 `ExecutionPlan` 为唯一核心路径：single-row 是薄封装，batch 是主路径，broadcast 复用同一套列式执行逻辑。broadcast 只允许复用显式 user-only 的中间结果，并补充 item-dependent 不复用、缺 item、空 item list、大 batch 测试。

### 3. JSON 请求解析需要 source-aware 类型转换

当前服务请求解析仍偏宽松，数组更容易被解析成 `StrList`，数值列表和配置中的 source dtype 没有强绑定；默认值解析也有静默兜底风险。

建议根据 `SourceDef.dtype` 做类型化 JSON 解析：`int`、`float`、`string`、`int_list`、`float_list`、`string_list` 分支明确。非法类型返回 `BadRequest`，默认值命中、类型转换失败、空序列进入指标和采样日志。

### 4. 序列 padding 需要上限保护

`feature_column_to_tensor()` 会使用批内观测最大长度或配置 `seq_len`。如果配置缺失且请求带来异常长序列，单次请求可能构造过大 tensor。

建议所有序列 embedding 必须有明确最大长度，或者服务层配置全局 `max_sequence_len`。超过上限时按策略截断、拒绝或降级，并记录 truncation count。

## P1：服务可靠性和发布机制

### 1. 建立结构化 API error

当前 `src/server/routes.rs` 的 `ErrorResponse` 只有 `error: String`，预测失败大多映射为 500。缺字段、类型错误、模型未加载、权重 key 不匹配、DAG 执行失败应有稳定错误码和不同 HTTP 状态。

建议引入统一 `ApiError`：包含 `code`、`message`、`request_id`、`model_id`、可选 `details`。错误至少分为 `BadRequest`、`FeatureError`、`ModelError`、`RegistryError`、`InternalError`，并在日志中使用结构化字段输出。

### 2. 拆分推理耗时统计

`src/server/routes.rs` 当前将整个 `engine.predict()` 计入 `dag_us`，而 `RequestTimer::record_model()` 没有被实际调用。线上排障时无法判断耗时来自 DAG、tensor 构建、模型 forward 还是结果序列化。

建议将 `InferenceEngine` 的执行拆成可观测阶段：request parse、DAG execute、feature tensor build、model forward、response build。`/predict` 和 `/predict/broadcast` 分别记录 batch size、item count、DAG 耗时、模型耗时和总耗时。

### 3. 模型注册中心需要完整上线生命周期

manifest 导出和校验已经具备基础能力，但 registry 仍缺少完整发布语义：候选加载、加载后 smoke predict、原子切换、回滚、模型信息查询和失败审计。

建议 `/models` 返回完整 `ModelInfo` 列表，包含 model id、version、feature config hash、model config hash、weights hash、loaded_at、status、last_error。加载流程采用 staged model，通过校验和 smoke predict 后再切换为 online。

### 4. 生产路径减少 `unwrap()`/`expect()`

`src/server/registry.rs` 等服务路径仍有锁、时间、文件名、hex 写入相关的 `unwrap()`。这些在正常情况下不常触发，但线上一旦触发会变成 panic，而不是可诊断错误。

建议服务路径统一返回带上下文的 typed error；`unwrap()`/`expect()` 只保留在测试、demo、不可恢复初始化路径，并用注释说明不变量。

## P2：训练数据、质量和算法能力

### 1. 补齐样本权重、负采样和偏差修正

当前训练配置和多任务 loss 已经抽象，但生产推荐训练还需要更完整的数据权重能力：

- 曝光位置偏差修正
- 负采样策略和采样权重回填
- 时间衰减权重
- 用户、物品、场景分桶采样
- 延迟转化窗口和 label 成熟期
- 任务级动态权重或不确定性权重

建议 batch 数据结构支持 `sample_weight`、`task_mask`、`context` 等字段，不要只传 features 和 labels。

### 2. 完善 FeatureHash 治理

基础 hash 分布统计已有训练侧能力，但 FeatureHash 的生产治理仍不完整。建议继续补齐：

- hash namespace / salt / version
- 重要字段的冲突率估算
- top raw value 到 bucket 的采样反查
- 低基数字段优先 DictMapper 的配置检查
- hash 分布进入训练报告、manifest 和线上监控

对于核心字段，训练报告应展示 hash 后分布，避免大字段被少量 bucket 异常支配。

### 3. 增加线上特征漂移监控

训练侧已有基础 feature quality 汇总，但还缺少训练集和线上请求之间的长期漂移监控。建议补齐：

- 每个 source 的线上缺失率、默认值命中率
- 每个 embeddable feature 的空序列率、序列长度分布
- 数值特征分位数、均值、方差、异常值比例
- 训练集和线上请求之间的 PSI/KL/分布漂移

这些指标应进入训练报告、模型 manifest 和线上监控，便于发现“模型没变但特征坏了”的问题。

### 4. 保持 Rust/Python 双端契约可验证

Python golden fixture 已存在，但仍需要把双端契约长期固化：YAML schema 版本、算子输入输出规范、pooling 和 sequence 语义、safetensors key 命名规范、不兼容变更策略。

新增算子或模型时，应同时提交 Python 实现、Rust 实现、契约文档和跨语言一致性测试。模型侧还需要固定同一 safetensors 权重、同一 batch 输入下 Python/Rust logits 的误差阈值。

## P3：Python 工程和配置生成链路

### 1. 训练入口继续收敛

`python/src/train/cli.py`、`python/src/train/main.py`、`python/src/scale_rec_demo/*.py` 已经比早期更清晰，但仍存在 demo 脚本和生产训练入口并行演进的风险。长期看，demo 应只负责组装样例参数，核心训练、评估、导出逻辑应只存在于 `train` package。

建议保留一个正式 CLI，demo 脚本只作为薄 wrapper 或文档样例。旧入口标记 deprecated，并在测试中覆盖正式 CLI 的最小训练、导出和加载。

### 2. 类型检查进入 CI baseline

Python 代码已有较多类型注释，但 `python/pyproject.toml` 目前只配置了 ruff 和 pytest 依赖，没有 `pyright` 或 `mypy` baseline。

建议先覆盖 `python/src/train/`，demo 目录可以晚一些纳入。初期不必全 strict，但要固定 baseline，新增代码不得扩大类型错误集合。

### 3. 配置生成器和 committed YAML 做结构化一致性测试

`examples/gen_discover_config.py` 已抽象生成逻辑，当前测试主要校验生成器的 contract 数量和 operator 名称唯一性，还没有比较生成结果与仓库 YAML 是否一致。

建议运行 generator 到临时路径，并与 committed config 做结构化 diff。若 generator 是唯一源头，则 CI 要求生成结果与仓库文件一致；若 YAML 允许手写，则文档要明确哪些字段不由 generator 管理。

### 4. 产物和缓存目录治理

代码树中仍容易出现 `__pycache__`、demo temp、生成 YAML、训练权重等本地产物。即使这些文件未进入 git，也会降低目录可读性。

建议统一 `.gitignore` 和目录约定：`python/demo/temp/`、`examples/generated/`、`artifacts/` 分别承担临时数据、生成配置和模型产物。文档中说明哪些文件可以删除、哪些是仓库源文件。

## P4：性能、测试和协议演进

### 1. 建立性能基线

建议为 Rust DAG plan、broadcast 推理、模型 forward 增加 criterion benchmark；Python 侧增加最小训练吞吐和 DAG batch preprocess benchmark。指标至少覆盖 p50/p95、batch size、序列长度、特征数量。

### 2. 补齐端到端失败用例

现有测试更多验证 happy path。应补充缺字段、错误类型、空序列、超长序列、unknown feature、权重缺 key、模型配置不匹配、broadcast item 缺失等失败用例，并验证错误码稳定。

### 3. 优化推理路径内存分配

Rust 特征工程路径中的 `Fv::Str(String)`、`Fv::IntList(Vec<i32>)`、`Fv::StrList(Vec<String>)` 会带来较多细粒度堆分配。

建议在性能敏感路径评估 `smallvec::SmallVec<[i32; 8]>`、`Cow<'a, str>` 或短字符串类型，减少短列表和短字符串的堆分配。

### 4. 增加推理设备抽象

Rust `ModelRegistry` 当前构建模型时固定使用 `Device::Cpu`。建议将 `Device` 抽离为启动配置项，为 Apple Silicon Metal、CUDA 或其他加速设备预留切换能力。

### 5. 评估 gRPC + Protobuf 协议

当前 Axum REST API 使用 JSON 传输大批量特征。高吞吐场景下 JSON 解析可能成为 CPU 瓶颈。中大规模部署可以评估 Protobuf 或 gRPC，降低带宽和反序列化成本。

## 建议执行顺序

1. 修 `ModelRegistry::load_model()` feature cache，并补充对应回归测试。
2. 引入结构化 `ApiError`，同步修正 source-aware JSON 解析。
3. 拆分推理耗时统计，并让 `/predict`、`/predict/broadcast` 分开记录。
4. 为配置生成器增加 committed YAML 结构化一致性测试。
5. 将 Python 类型检查加入 CI baseline。
