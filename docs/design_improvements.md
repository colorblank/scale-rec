# scale-rec 设计分析与改进方案

本文基于当前代码库实现，从架构、功能、性能和工程治理角度梳理现状、风险与改进路线。目标是让后续优化可以按优先级拆分为明确任务，而不是停留在笼统建议。

## 1. 当前架构概览

scale-rec 采用 Python 训练 + Rust 推理的双运行时架构：

- Python 侧使用 PyTorch、pandas/Polars 相关数据处理和 safetensors 导出，主入口集中在 `python/src/train/app/main.py` 与 `python/src/train/training/trainer.py`。
- Rust 侧使用 Candle 承载模型推理，HTTP 服务由 `src/server` 提供，核心推理链路在 `src/server/engine.rs`。
- 双端共享同一份 YAML 特征配置，由 Python `python/src/train/core/config.py` / `python/src/train/core/dag.py` 与 Rust `src/feats/config.rs` / `src/feats/dag.rs` 解析。
- 训练任务语义已经从模型代码里抽出，使用 `tasks`、`label_col_map`、`metrics` 在模型 YAML 里声明，`Trainer` 和 `Evaluator` 只读取配置，不再按模型类型猜测监督目标。
- 模型特征规格不在模型配置中重复声明，而是统一来自 `FeatureDag.embeddable_features()`。
- Python 训练后导出 safetensors，Rust 通过 Candle `VarMap` 加载；权重 key 必须与 Rust `VarBuilder::pp()` 路径严格对齐。
- `examples/` 已拆分为 `examples/shared/` 和 `examples/models/`，共享配置与模型配置不再混放；当前模型示例包括 `lr.yaml`、`gdcn_esmm.yaml`、`unimixer.yaml`。
- 训练和推理的全链路验证已经收敛到 `python/src/scale_rec_demo/verify_all.py`，并覆盖 `discover_lr`、`discover_gdcn_esmm`、`discover_unimixer` 三条主线；最近一次实测已通过 Python 训练、safetensors 导出、Rust 推理和输出比对四段链路。
- Python 训练入口已经抽出公共批处理 helper `iter_preprocessed_batches()`，`single` / `discover` / `all` 三条入口共享同一套批次预处理与可选预取逻辑。
- 训练器已经支持周期 checkpoint 保存，可按步数或时间间隔在 epoch 中途落盘，而不是只依赖 epoch 结束。

这套设计的核心优点是：特征编排有单一来源，训练与推理模型结构可对齐，Rust 线上服务可以避免携带 Python 运行时。同时，当前代码已经具备多模型注册、发布 manifest、Golden consistency、模型 smoke test、算子级测试、训练/推理一致性脚本和任务级配置。

## 2. 架构分析

### 2.1 错误处理与系统鲁棒性

当前代码库在生产路径上大量使用不可恢复的错误处理方式：

- **`unwrap()` / `expect()` 泛滥**：全库约 39 处 `.unwrap()` / `.expect()` 调用分布在生产路径上（不含测试代码），其中 14 处为高严重度。
  - `src/main.rs:21-22`：配置文件读取和 YAML 解析直接 `.expect()`，文件不存在或格式错误导致进程崩溃。
  - `src/bin/server.rs:179-180`：`TcpListener::bind().await.unwrap()` 和 `axum::serve().await.unwrap()`，端口被占用或服务异常直接 panic。
  - `src/bin/server.rs:36,131,133`：Tokio runtime 构建、model registry 创建均 `.expect()`。
  - `src/feats/ops/expression.rs:14`：Rhai 表达式编译失败直接 `.expect("Invalid Rhai expression")`。
  - `src/server/registry.rs:279,295,418,436,442,452,484,605`：多处 `RwLock` 的 `.lock().unwrap()`，在 `Mutex`/`RwLock` 已 poisoned 时会导致级联 panic。
  - `src/layers/embedding.rs:116,129`：`HashMap::get().unwrap()` 在 feature index 查不到时 panic。
- **`panic!()` / `assert!()` 在非测试代码中**：
  - `src/main.rs:75`：特征类型不支持时 `panic!("Feature '{}' has unsupported type", ...)`。
  - `src/feats/ops/feature_hash.rs:36`：`assert!(vocab_size > 0, "vocab_size must be positive")`。
- **静默吞错**：
  - `src/models/mod.rs:333`：`serde_yaml::from_value::<MultiTaskConfig>().unwrap_or_else(|_| MultiTaskConfig { towers: vec![], relations: vec![] })` —— 配置错误时返回零 tower，服务启动无任何报错。
  - `src/feats/defaults.rs:12-25`：`parse_int_strict(raw).unwrap_or(0)` 对非法默认值静默回退为 0。
  - `src/feats/ops/expression.rs:29`：非数值输入静默转为 0.0。
  - `src/feats/ops/plugin.rs:42`：插件返回不识别类型时 `.unwrap_or(Fv::Int(0))`。
  - `src/feats/dag.rs:195,200`：使用 `usize::MAX` 作为 sentinel 值，下游使用错误的值会导致难以追踪的 bug。
  - `src/feats/debug/tracer.rs:235,244,247,257-262`：文件 I/O 错误用 `let _ = ` 静默丢弃。
- **Python 侧 `except Exception` 吞错**：
  - `python/src/train/app/data.py:78`：`_read_file_compat` 中 `except Exception` 捕获所有异常后 fallback 到 relaxed 模式，MemoryError/KeyboardInterrupt 等关键异常也被吞掉。
  - `python/src/train/app/manifest.py:30`：`current_git_commit` 中 `except Exception` 返回 `"unknown"`，不留任何日志。

**改进方向**：
- 将生产路径（`src/bin/`、`src/server/`）所有 `.unwrap()`/`.expect()` 替换为 `Result` 传播，配合 `tracing::error!` 记录上下文。
- 将 `panic!()` 和 `assert!()` 在非测试代码中替换为 `Result::Err` 或 `tracing::error!` + graceful degradation。
- 为 `serde_yaml` 反序列化增加 `#[serde(deny_unknown_fields)]`，防止配置拼写错误被静默忽略。
- Python 侧 `except Exception` 替换为具体异常类型，至少记录 `logging.exception()` 后再 fallback。
- Rust 侧所有默认为 0/Int(0) 的 `.unwrap_or()` 至少添加 `tracing::warn!`，关键位置改为 `Result::Err`。

### 2.2 日志规范

当前代码中 `println!()` / `eprintln!()` / `print()` 大量散布在生产和非测试代码中，完全绕过 tracing 日志体系：

- **Rust**：约 30 处 `println!()` 分布在 9 个文件中：
  - `src/main.rs:19-133`：14 处 println，包括配置读取、执行结果、输出展示。
  - `src/feats/dag.rs:302-322`：`validate()` 用 println 输出 `[DAG] sources:`、`[DAG] WARNING` 等信息，生产者无法捕获。
  - `src/bin/server.rs:77,89`：错误信息用 `eprintln!`。
  - `src/feats/metrics.rs:45-60`：指标报告用 println。
- **Python**：9 处 `print()`：
  - `python/src/train/app/export.py:23`：`print_state_dict_keys` 用 print 而非 logging。
  - `python/src/scale_rec_demo/verify_all.py:57-226`：6 处 print。

**改进方向**：
- 全量替换为 `tracing::info!/warn!/error!`（Rust）和 `logging.info/warning/error`（Python）。
- 确保 log level 可通过环境变量或配置控制，生产默认不低于 INFO。

### 2.3 FeatureDag 与特征配置

当前特征系统的边界比较清晰：

- `FlowConfig` 描述 sources、operators、embed 配置。
- `FeatureDag` 完成拓扑排序、算子创建、schema 推导、验证报告和执行。
- Python `FeatureDag.preprocess_batch()` 负责训练侧从 batch rows 到模型输入 tensor。
- Rust `FeatureDag` 同时保留单样本 HashMap 执行路径和批量预编译 plan 路径。
- 当前特征 DAG 已经支持两个融合算子：`ParsedFeatureHash` 和 `ConcatHash`，用于把“解析 + hash”合并成单节点，减少中间值和配置层级。

优势：

- YAML 配置是训练和推理共享契约，减少了特征漂移风险。
- Rust 侧已有 `ExecutionPlan`，将算子输入输出预解析为列 id，降低批量推理中的 HashMap 查找开销。
- schema 推导已经开始下沉到 DAG 层，后续可以继续作为运行时契约来源。
- DAG 构建阶段已经有 orphan source/output 检查，能提前发现部分配置问题。

主要风险：

- Rust 单样本执行路径与 plan 批量路径并存，默认值填充、缺失列处理、错误信息可能出现语义差异。
- 默认值解析在多个位置重复存在，例如 `FeatureDag::parse_default()`、`source_default()`、Python `_parse_default()`，长期容易产生训练/推理不一致。
- `execute_batch_precomputed()` 中仍有旧 HashMap 批量路径，与 plan 路径职责重叠，增加维护成本。
- 当前 schema 还没有完全成为唯一执行契约，部分逻辑仍依赖算子本地默认值、字符串判断或 fallback；尤其是融合算子的 parse mode 和 embedding seq_len 推导，仍需要在 Python/Rust 两边保持严格一致。

改进方向：

- 将 plan 执行路径确立为 Rust 推理主路径，逐步减少或隔离旧批量执行路径。
- 抽出统一默认值解析逻辑，Rust 侧避免 `parse_default()` 与 `source_default()` 两套行为。
- 将 `FeatureSchema` 的 dtype、seq_len、pooling、default、role 作为训练和推理的统一契约。
- 对 DAG validation 增加严格模式，生产加载模型时可以选择 orphan warning 升级为错误。

### 2.4 模型与权重加载

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
- `tasks` / `label_col_map` / `metrics` 已经统一进模型配置和 manifest，训练、导出和 serving 都可以直接读取。
- `task_config` 负责 tower、relation 和前向结构；`tasks` 负责监督、损失和评估。两者职责已经分开，但文档和代码注释里仍要持续强调边界，避免新的参数重新耦合。

改进方向：

- 为每个模型补充自动化 state_dict key 对齐测试，尤其是新增层或新增任务时。
- 收敛 legacy hidden_dims 参数，长期以 `task_config` / `tasks` 作为多任务配置主入口。
- 在 manifest 中记录模型输出任务、label、relation 和训练指标，并让服务端 `/models` 返回关键元数据。

### 2.5 训练产物与发布

训练侧已有 `TrainingArtifactManager`：

- run 目录保存 checkpoints、best/latest alias、run manifest 和默认 serving 发布目录。
- 默认发布路径保存到当前 run 的 `serving/model.safetensors` 和 `serving/model.manifest.yaml`，配置副本保存到 `serving/configs/`。
- manifest 记录 weights、feature config、model config 的 sha256。

这是正确方向，已经解决了“只有权重文件无法恢复训练上下文”的问题。

另外，训练侧已经支持周期 checkpoint：可以按 `checkpoint_interval_steps` 或 `checkpoint_interval_seconds` 在训练中途保存，并用 `periodic-*` 版本名与普通 epoch-end checkpoint 区分。

当前不足：

- `copy_configs` 已默认开启，默认发布产物会在 `serving/configs/` 归档 feature/model config；但如果显式把 `--publish-path` 指到 run 目录外，跨机器部署时仍要同时携带 run 目录里的配置副本。
- Rust `ModelRegistry` 已支持按 serving manifest 的 `feature_config_file`、`model_config_file`、`weights_file` 加载，并支持同一 `model_id` 多版本；服务已能按模型版本暴露 feature config 中的 `data_sources` 和请求输入字段契约，但 `/models` 还没有把 `tasks`、`label_col_map` 和 `metrics` 完全暴露为查询接口。
- 默认版本目前按版本字符串取最大值，尚未支持显式 alias、灰度权重或 `versions.yaml` 指针。

改进方向：

- 保持生产发布产物自包含，避免 serving manifest 依赖仓库 `examples/` 等外部配置路径。
- 扩展 `ModelInfo`，返回 schema hash、tasks、metrics 和 loaded_at；特征契约继续通过 `/models/{model}/features` 查询。
- 增加显式默认版本配置，例如 serving manifest 标记、alias 文件或版本索引文件。

### 2.6 工程基础设施

当前项目工程基础设施在以下方面存在明显缺口：

**CI/CD 管线完全缺失**：无 `.github/workflows/`、无 `.gitlab-ci.yml`、无 Jenkinsfile。测试、lint、构建和部署全依赖人工执行，缺少自动化质量闸门。

**Cargo.lock 已纳入仓库，但仍需要 CI 约束**：
- `Cargo.lock` 已被跟踪，Dockerfile 也按 `cargo build --locked` 构建。
- 这条链路已经从“全新 clone 直接失败”修复到“依赖必须受锁文件约束”。
- 后续风险主要是有人在本地绕过锁文件或错误地回退提交，所以 CI 仍需要强制 `--locked`。

**静态检查工具配置不完整**：
- Rust：无 `rustfmt.toml`、`clippy.toml`，代码风格仅靠默认规则 + `.claude/rules/code-style.md`（LLM 指导），无机械执行。
- Python：`ruff` 仅启用默认 pycodestyle + pyflakes，缺少 `I`(isort)、`D`(pydocstyle)、`B`(flake8-bugbear)、`UP`(pyupgrade)、`SIM`(flake8-simplify) 等规则集。
- **Mypy 种类检查形同虚设**：`python/pyproject.toml:50-72` 对 22 个核心模块（含 `train.core.config`、`train.core.dag`、`train.models` 等）设置 `ignore_errors = true`。
- 无 `.editorconfig`，编辑器间换行/缩进/编码可能不一致。

**依赖安全扫描缺失**：无 `cargo-audit`、`cargo-deny` 配置，无 Dependabot/Renovate。14 个 Rust crate 加上 Python 依赖无自动漏洞检查。

**无 pre-commit hooks**：格式化和 lint 完全依赖开发者自觉，或 CI 中事后发现。

**改进方向**：
- 在 CI 中强制 `cargo build --locked` / `cargo test --locked`，避免锁文件被绕过。
- 搭建 GitHub Actions（或 GitLab CI）：包含 `cargo test/fmt/clippy`、`ruff check/format`、`mypy`、`pytest`、`cargo-audit`。
- 扩展 ruff 配置加入 `I`/`B`/`UP`/`SIM` 规则集，渐进式修复后逐步解锁 mypy `ignore_errors`。
- 添加 `.editorconfig` 和 `.pre-commit-config.yaml`。
- 配置 `cargo-deny` 并集成到 CI。

### 2.7 安全与防御

**服务端安全**：
- `src/bin/server.rs:175`：`CorsLayer::permissive()` 允许任意跨域请求，任何网站都能调用推理接口。
- 无速率限制 middleware，无请求体大小上限，单个客户端可 OOM 整个服务。
- 所有 HTTP 端点无认证，任何人可访问 `/predict`、`/models` 等。
- `src/server/routes.rs:56-72`：`ApiError::into_response()` 可能向客户端暴露内部路径和错误详情。

**不安全代码**：
- `src/feats/ops/plugin.rs:18,53-59`：`unsafe { Library::new(path) }` 加载任意本地 `.so/.dll`，无路径校验和签名验证，攻击者可通过控制 `path` 执行任意代码。缺少 `// SAFETY:` 注释说明前置条件。
- `src/server/registry.rs:609`：`unsafe { MmapedSafetensors::new(path) }` 载入外部文件，同样缺少安全注释。

**Docker 安全**：
- Dockerfile 无 `USER` 指令，容器以 root 运行。
- `src/bin/bench.rs:14`：硬编码 `http://localhost:8080` 默认 URL（开发级风险）。

**改进方向**：
- CORS 改为白名单模式，仅允许可信 origin。
- 添加 `tower::limit::RateLimitLayer` 和 `tower_http::limit::RequestBodyLimitLayer`。
- 添加 API key 或 mTLS 认证（至少先在内部部署边界完成）。
- 为所有 unsafe 块添加 `// SAFETY:` 注释，插件路径增加白名单校验。
- Dockerfile 添加 `USER 1000`。

### 2.8 公共 API 文档覆盖率

当前关键公共类型和方法缺少 rustdoc 或 docstring：

- **Rust**（15 项）：`FeatureDag`、`ExecutionPlan`、`ExecStep`、`ModelConfig`、`TaskConfigEntry`、`InferenceEngine`、`FeatureRow`、`PredictionRow`、`Fv`、`CustomOp`、`ModelRegistry`、`FeatureSchema`、`FeatureDType`、`FeatureSpec`、`FeatureEmbeddings` 均无或仅有最基础的 doc comment。
- **Python**（约 55 项）：`FeatureDag` 类及其所有方法、`Trainer` 类及其所有方法、`MultiTaskLoss`、cli 模块所有函数、`FlowConfig`、`ModelConfig`、分类指标函数、`verify_all` 中的公共函数等均无 docstring。

**改进方向**：
- 为所有 `pub` 类型和 `pub fn` 添加至少一行文档描述用途。
- 对关键路径（DAG、模型注册、训练器）补充示例代码。
- 在 CI 中启用 `#[warn(missing_docs)]`（Rust）和 `D` 规则（ruff pydocstyle），逐步修复。

### 2.9 代码重复与 God Class

**代码重复**：
- `python/src/train/app/main.py` 与 `python/src/train/training/trainer.py` 仍各自承担一部分 raw batch 组织和预处理入口 glue 逻辑，虽然核心预处理已收敛到 `iter_preprocessed_batches()`，但入口层的批次结构理解仍可继续统一。
- `src/feats/dag.rs:534-563`：`op_source_kind()` 包含两段几乎相同的逻辑计算 `k` 值，存在漂移风险。
- `python/src/train/app/main.py:43` 和 `python/src/train/app/data.py:17`：`NULL_MARKERS` 常量重复定义。

**God Class（违反单一职责原则）**：
- `python/src/train/core/dag.py` 的 `FeatureDag`（~485 行）：同时负责 DAG 构建、拓扑校验、单样本执行、批量执行、tensor 预处理、特征元数据提取和调试。
- `python/src/train/training/trainer.py` 的 `Trainer`（~503 行）：同时负责训练循环、数据迭代、评估、EMA、checkpoint 管理、artifacts 和日志。

**改进方向**：
- 继续收敛 Python 训练入口中的 batch 组织逻辑，避免 `main.py` 与 `trainer.py` 对同一批次结构维护两套认知。
- 将 `NULL_MARKERS` 抽取到公共模块（如 `train.constants`）。
- 将 `FeatureDag` 拆分为 `DagBuilder`（构建+校验）、`DagExecutor`（执行）、`DagPreprocessor`（tensor 转换）。
- 将 `Trainer` 拆分为 `EMA`、`CheckpointManager`、`TrainingLoop` 独立组件。

## 3. 功能分析

### 3.1 已具备能力

当前代码库已经具备以下核心功能：

- 共享 YAML 特征编排，覆盖多种 operator。
- 已新增融合预处理算子 `ParsedFeatureHash` / `ConcatHash`，把高频解析+hash链路合并，减少 DAG 深度和中间值。
- Python 单文件训练、discover 训练、多模型训练入口，且三条入口共享同一个批处理预处理/预取 helper。
- 多任务 loss、评估、early stopping、EMA、周期 checkpoint、epoch-end checkpoint 和 manifest。
- 任务级配置已经落到模型 YAML，训练和评估直接读 `tasks`、`label_col_map`、`metrics`。
- Rust HTTP 推理，支持 pointwise `/predict` 和 broadcast `/predict/broadcast`。
- Rust 推理侧支持 batch DAG、broadcast user 子图预计算和模型热加载。
- Golden consistency 测试覆盖 Python/Rust 特征处理一致性。
- safetensors 权重 key 与 shape 校验。
- benchmark 工具支持 synthetic 和 discover 输入压测。

这些能力说明项目已经从 demo 形态进入“可工程化迭代”的阶段，下一步重点不是堆功能，而是收紧契约、降低漂移风险和提升吞吐稳定性。

### 3.2 功能缺口

需要优先补齐的功能缺口：

- Typed error：当前 `src/server/routes.rs` 的 `map_predict_error()` 仍依赖字符串包含关系分类错误，未来错误文案变化会导致 HTTP status 不稳定。
- 模型级 schema：registry 已按 manifest 加载多模型 feature config，并能查询请求特征契约；后续还需要暴露 embeddable schema、schema hash 和兼容性检查结果。
- 配置兼容策略：feature config、model config、manifest schema 的版本兼容规则尚未系统化。
- 线上可观测性：已有 parse/dag/tensor/forward/response 耗时，但缺少 feature default hit rate、空序列比例、截断次数、batch size、broadcast item count 等指标。
- 数据质量闭环：训练侧有 feature quality summary，但还没有与 manifest、线上日志和服务端统计形成统一链路。
- **错误处理体系**：生产路径 39 处 `.unwrap()`/`.expect()` 散布在 server、DAG、feature hash、registry、embedding 等模块中，任何一处触发都会导致进程崩溃。
- **日志体系**：30 处 `println!()` 分布在核心路径（DAG validation、推理主循环、metrics 报告），无法被日志系统采集和路由。
- **God Class 重构**：`FeatureDag`（~485 行）和 `Trainer`（~503 行）职责过多，后续加功能会继续膨胀。
- **安全防护**：HTTP 服务无 CORS 白名单、无速率限制、无认证、无请求体大小限制。
- **CI/CD 自动化**：完全依赖人工执行 `cargo test`、`pytest`、`ruff`、`cargo fmt`。
- **类型检查**：Mypy 对 22 个核心模块设置为 `ignore_errors = true`，实际未生效。
- **训练中断恢复**：周期 checkpoint 和 `--resume-from` 已支持完整训练状态恢复；后续更值得补的是自动选择最近可恢复 checkpoint、以及更细粒度的训练中断回放能力。

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
- `ParsedFeatureHash` / `ConcatHash` 已经把多段链路合并，减少了解析和哈希之间的中间对象分配。

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
- 训练开始会打印数据摘要，包括总行数、train/eval 切分、batch 估算、任务和标签映射，便于快速定位“无监督 batch”或标签列不匹配问题。

性能风险：

- pandas chunk 到 dict records 再到 columns 属于重复转换，CPU 和内存开销都较高。
- `build_item_index()` 虽然已用 `itertuples()`，但最终仍构造大字典，物品规模很大时内存压力明显。
- 训练数据 pipeline 现在已经有线程池 prefetch，但主路径仍然是 pandas chunk + `to_dict("records")` + DAG 预处理；预取只能隐藏一部分 CPU 开销，不能消除数据结构转换的成本。
- Python 和 Rust 的 batch DAG 行为仍需持续用 Golden 测试锁住，否则优化过程中容易产生细微差异。

改进方向：

- 让 `stream_file_batches()` 直接 yield `dict[str, list]` 列式 features，避免 `to_dict("records")`。
- `FeatureDag.preprocess_batch()` 优先走列式输入，list[dict] 作为兼容路径。
- 大数据训练引入 Polars LazyFrame 或 Arrow RecordBatch 作为中间格式。
- 在现有 prefetch 之上继续拆分读取、预处理和设备拷贝，形成真正的 producer/consumer 流水线。
- 对关键算子补充 batch consistency 测试：单样本执行与 batch 执行输出必须一致。

## 5. 改进优先级

### P0：稳定性与构建可复现（立即修复）

1. **保持 `Cargo.lock` 入仓并在 CI 中强制 `--locked` 构建**，防止依赖漂移和 Docker 复现问题回归。
2. **用 `Result` 替换生产路径所有 `.unwrap()`/`.expect()`**：
   - `src/bin/server.rs:36,131,133,179,180`：服务启动和路由绑定。
   - `src/feats/dag.rs:167,187`：DAG source/op 查询。
   - `src/server/engine.rs:195,316,379`：预测路径 source output 查询。
   - `src/layers/embedding.rs:116,129`：feature index 查询。
   - `src/feats/ops/feature_hash.rs:61,141,162,169`：缓存 RwLock。
   - `src/server/registry.rs:279,295,418,436,442,452,484,605`：模型注册 RwLock。
3. **用 `tracing` 替换所有 `println!()`/`eprintln!()`**（仍然较多）：让日志可被采集、路由和级别控制。
4. **修复 Python `except Exception` 吞错**：
   - `python/src/train/app/data.py:78`：`_read_file_compat` 改为 `except (pd.errors.ParserError, ValueError)`。
   - `python/src/train/app/manifest.py:30`：`current_git_commit` 至少记录 `logging.exception`。
5. **为 `serde` struct 添加 `#[serde(deny_unknown_fields)]`**：防止 YAML 配置拼写错误被静默忽略。
6. 保持 registry 的多模型独立 schema 模式：按 serving manifest 的 `feature_config_file` 构建 DAG，无 manifest 的旧权重仅作为开发 fallback。
7. 引入 typed inference error，替换 `map_predict_error()` 的字符串匹配。
8. 统一 Rust 默认值解析逻辑，避免 `FeatureDag::parse_default()` 与 `source_default()` 行为分叉。
9. 为 Python/Rust batch DAG 增加更多一致性测试，尤其是 list、flatten、null/default、FeatureHash，以及融合算子的 parse mode。

### P1：安全与工程治理（近期）

1. **CORS 改为白名单**：`src/bin/server.rs:175` 的 `CorsLayer::permissive()` 替换为限制性配置。
2. **添加速率限制和请求体大小限制**：使用 `tower::limit::RateLimitLayer` 和 `RequestBodyLimitLayer`。
3. **添加 API 认证**：至少先在内部部署边界完成 API key 或 mTLS。
4. **为 unsafe 块添加 `// SAFETY:` 注释**：
   - `src/feats/ops/plugin.rs:18,53-59`：动态库加载与 FFI 调用，同时增加路径白名单。
   - `src/server/registry.rs:609`：`MmapedSafetensors::new`。
5. **搭建 CI/CD 管线**（GitHub Actions）：含 `cargo test/fmt/clippy`、`ruff check/format`、`mypy`、`pytest`、`cargo-audit`。
6. **扩展 ruff 配置**：加入 `I`/`B`/`UP`/`SIM` 规则集，逐步修复后解锁 mypy `ignore_errors`。
7. **添加 `.editorconfig` 和 `.pre-commit-config.yaml`**。
8. **修复静默吞错**：
   - `src/models/mod.rs:333`：`unwrap_or_else` 改为 `map_err` + 日志。
   - `src/feats/defaults.rs:12-25`：非法默认值添加 `tracing::warn!`。
   - `src/feats/ops/expression.rs:29`：非数值输入添加 `tracing::warn!`。
9. **消除代码重复**：
   - 继续收敛 Python 训练入口中的 batch 组织逻辑，减少 `main.py` 与 `trainer.py` 对同一批次结构的重复理解。
   - 将 `NULL_MARKERS` 抽取到公共模块。
10. Dockerfile 添加 `USER 1000`。

### P2：性能与可观测性（中期）

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
6. 在 tensor 构造阶段减少中间 clone（`src/feats/dag.rs:593,617-621,679,684,805`、`src/server/engine.rs:304-306`）。
7. 收敛模型多任务配置入口，逐步减少 legacy hidden_dims 参数路径。

### P3：重构与平台化（长期）

1. **拆分 God Class**：
   - `FeatureDag` → `DagBuilder` + `DagExecutor` + `DagPreprocessor`。
   - `Trainer` → `EMA` + `CheckpointManager` + `TrainingLoop`。
2. 为所有公共 API 补充文档（15 项 Rust pub 类型、55 项 Python 函数/类）。
3. 支持列式请求 API 或二进制协议，服务高吞吐召回/排序场景。
4. 将 operator 注册机制标准化，降低新增算子的 Rust/Python 双端维护成本。
5. 对大规模训练引入 Arrow/Polars-first pipeline，并把现有 prefetch 演进为真正的 producer/consumer 流水线。
6. 为线上推理增加 Prometheus/OpenTelemetry 指标导出。
7. 建立模型发布、回滚、灰度和兼容检查流程。
8. 让生产发布产物默认自包含 feature config 和 model config，并保持特征契约查询接口只读取发布归档配置。
9. 为模型 state_dict key 对齐建立自动化测试或导出检查脚本。
10. 将训练侧 feature quality 写入 manifest，并在服务端加载后可查询。

## 6. 推荐执行路线

**短周期（P0，1-2 周）**：保持 Cargo.lock 入仓、消除生产路径 unwrap/expect、引入 tracing 替换 println、修复 Python except Exception、添加 deny_unknown_fields。核心目标是消除进程崩溃风险，让错误能够传播和追踪。

**短中期（P1，3-4 周）**：搭建 CI/CD、修复安全漏洞（CORS/限流/认证/unsafe）、扩展 lint 工具链、修复静默吞错。核心目标是建立质量闸门，让安全问题不再被忽略。

**中期（P2，1-2 月）**：列式训练数据流、FeatureHash 可观测性、InferenceMetrics 扩展、tensor clone 优化。核心目标是在有基线的前提下提升吞吐。

**长期（P3，2-4 月）**：重构 God Class、补充公共 API 文档、Arrow pipeline、Prometheus 指标导出、模型发布治理。核心目标是把当前工程化能力提升为稳定平台能力。

## 7. 验收建议

每一轮改进至少满足以下验收条件：

- Rust：`cargo check`、`cargo test`、`cargo fmt`、`cargo clippy` 通过。
- Python：`PYTHONPATH=python/src uv run pytest python/tests/ -v`、`uvx ruff check python/src/`、`uvx ruff format python/src/`、`uv run mypy python/src/` 通过。
- CI：所有检查在 CI 中自动执行，PR 合并前必须通过。
- 格式：Rust 使用 `cargo fmt`，Python 使用 `uvx ruff format python/src/`。
- 一致性：涉及特征、算子、模型结构或权重命名时，必须跑 Golden consistency 或对应 verify 脚本。
- 性能：涉及推理或训练性能时，必须记录优化前后的 batch size、模型、后端、P50/P95/P99、RPS 或 per-batch timing。
- 安全：CORS 必须为白名单；速率限制和请求体大小限制必须存在。
- 文档：新增公开 API 必须包含 doc comment / docstring。

总体建议是先收紧契约和修复安全漏洞，再优化性能，最后平台化。当前代码库已经具备较好的分层基础，后续改进应避免大范围重写，优先在现有 `FeatureDag`、`ModelRegistry`、`Trainer`、manifest 体系和融合算子上增量演进。
