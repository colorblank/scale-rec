# scale-rec 设计分析与改进方案

本文基于当前代码库实现更新，目标是把已完成的基础治理项从待办中移出，并把后续优化收敛到仍然影响生产稳定性、安全性、性能和演进效率的真实问题上。

## 1. 当前架构概览

scale-rec 采用 Python 训练 + Rust 推理的双运行时架构：

- Python 侧位于 `python/src/train`，负责数据读取、特征预处理、PyTorch 训练、评估、checkpoint、artifact 管理和 safetensors 导出。
- Rust 侧位于 `src`，负责共享特征配置解析、Candle 模型推理、HTTP serving、多模型 registry、manifest 加载和压测工具。
- 双端共享同一份 YAML 特征配置。Rust 通过 `src/feats/config.rs`、`DagBuilder`、`DagExecutor`、`FeatureInfo` 解析、校验、执行和暴露特征元数据；Python 通过 `python/src/train/core/config.py`、`DagBuilder`、`DagExecutor`、`FeatureInfo`、`DagPreprocessor` 镜像实现，`FeatureDag` 仅保留为兼容 facade。
- 特征 source 已支持 `role = feature | label | discard`，训练侧按 role 分离 feature/label/discard，推理侧只暴露 feature 输入契约。
- 模型特征规格统一来自 `FeatureInfo.embeddable_features()`，模型配置不重复声明特征清单。
- 全部 8 个注册模型已支持 `output_contract.version: 1`，统一描述 tower、relation、
  objective、metric 和公开输出；legacy 字段仅保留兼容性。
- Python 训练后导出 safetensors，Rust 通过 Candle `VarMap` 加载。权重 key 必须与 Rust `VarBuilder::pp()` 路径严格对齐。
- Rust serving 以 manifest 为主路径加载模型，支持同一 `model_id` 多版本、alias、固定/加权 routing、按版本查询 feature contract。
- HTTP 服务提供 `/health`、`/models`、`/models/{model}`、`/models/{model}/features`、`/models/{model}/versions/{version}/features`、alias/routing 管理、`/predict`、`/predict/broadcast`。
- Python 训练入口已经统一到列式 batch 预处理路径，`stream_file_batches()` 直接产出 `dict[str, list]`，训练/评估/metrics 使用 `TrainingPreprocessor`，底层通过 `DagPreprocessor` 完成 tensor 构造。
- `examples/` 已按 `examples/shared/` 和 `examples/models/` 拆分，全部模型示例使用原生
  输出契约。
- 训练和推理一致性验证集中在 `python/src/scale_rec_demo/verify_all.py`，覆盖全部 8 个
  注册模型。

这套设计的核心优势是：特征契约有单一来源，训练与推理模型结构可对齐，线上服务不携带 Python 运行时，同时已经具备多模型注册、发布 manifest、Golden consistency、模型 smoke test、算子级测试、周期 checkpoint、resume、压测工具和任务级配置。

### 1.1 通用模型结构与业务特定模型的关系

模型系统应按三层理解，而不是把每个 `type` 都视为同一层级的业务模型：

- **通用模型结构层**：提供可复用的表示学习和输出结构，例如 `FeatureEmbeddings`、`FeatureTokenizer`、`Mlp`、`GatedCrossNetwork`、UniMixer/RankMixer block、`TaskTower`、`MultiTaskTower`、`TaskRelation` 和 `ModelOutput`。这一层关心张量形状、权重命名、输出语义和双端一致性，不应携带 discover 业务字段名或标签策略。
- **任务契约层**：原生路径用 `output_contract` 统一描述 typed graph、objective、
  metric 和 public output。legacy 路径继续兼容旧配置，两条路径禁止混用。
- **业务特定模型层**：把通用结构和任务契约组合成某个业务场景可用的模板，例如 discover 场景的 `gdcn_esmm.yaml`、`rankmixer.yaml` 中的 click/cvr/detail/stock/stay tower、ctcvr/ctdetail/ctstock/ctstay relation、标签列名和指标。业务层应该主要体现在 YAML 配置、label policy、artifact manifest 和发布策略中，而不是散落到模型基础实现里。

因此，`gdcn_esmm`、`rankmixer`、`token_mixer_large` 这类模型 `type` 更准确地说是“结构模板”：它们决定 shared representation 如何产生、是否使用 token mixer、是否使用 cross/deep/fusion，但不应该固定某个业务的任务集合。click/cvr/detail/stock/stay 是 discover 业务的一组任务契约；当新业务接入时，理想路径应是复用同一个结构模板，替换 `output_contract`，而不是复制一个新模型文件并把业务名写进代码。

重新梳理后的架构边界如下：

```text
raw sample / request
  -> FlowConfig + DagBuilder + DagExecutor
  -> FeatureInfo.embeddable_features()
  -> generic structure template: lr/deepfm/gdcn_esmm/unimixer/rankmixer/...
  -> output_contract v1 (legacy tasks/task_config compatibility)
  -> ModelExecution: internal typed nodes + public ModelOutput
  -> training loss/metrics or serving score selection
```

这个边界带来的判断规则：

- 如果变化是“特征如何从原始字段生成”，放在 feature config/operator/DAG。
- 如果变化是“张量结构如何表达特征交互”，放在通用模型结构或新增结构模板。
- 新模型把 tower、概率关系、loss、metric 和公开输出放在 `output_contract`。
- legacy 字段只用于旧配置兼容，不再作为新接入方式。
- 如果变化是“线上用哪个输出排序、版本如何切流”，放在 serving manifest、alias/routing 或业务排序策略，不放进基础模型实现。

当前代码已经新增双端 `OutputContract` 校验、`OutputHead`、`ModelExecution` 和 Python
`ObjectiveEngine`，并完成全部 8 个注册模型及示例配置迁移。manifest 还没有保存规范化
契约，因此下一阶段重点是完善发布元数据；legacy 路径暂不删除。

## 2. 已完成的关键治理项

相比上一版分析，以下问题已经明显改善，后续文档和计划不应再把它们作为未完成 P0：

- **Server 启动错误传播**：`src/bin/server.rs` 的 Tokio runtime、registry 创建、TCP bind 和 `axum::serve` 已使用 `Result` + `anyhow::Context`，不再用启动路径 `.unwrap()` 直接崩溃。
- **Typed inference error**：`src/server/engine.rs` 已定义 `InferenceErrorKind`，`src/server/routes.rs` 的 `map_predict_error()` 通过错误类型映射 HTTP status，不再依赖字符串包含关系。
- **Rust 日志体系**：`src` 下生产代码基本已从 `println!()` / `eprintln!()` 迁移到 `tracing`，server 支持 `RUST_LOG` 控制日志级别。
- **CORS 白名单**：server 不再使用 `CorsLayer::permissive()`，默认只允许本地开发 origin，并支持 `SCALE_REC_ALLOWED_ORIGINS` / `--allowed-origin` 配置。
- **请求体大小限制**：server 已通过 `DefaultBodyLimit::max(args.max_body_bytes)` 限制请求体，默认 8 MiB，并支持 `SCALE_REC_MAX_BODY_BYTES` / `--max-body-bytes`。
- **Rust feature config 和 serving manifest 严格反序列化**：`FlowConfig`、`SourceDef`、`OperatorDef`、`EmbedConfig`、`DataSourceDef`、`ModelManifest`、`WeightBinding` 已添加 `#[serde(deny_unknown_fields)]`。
- **基础过载保护**：server 已支持可配置的全局 rate limit、并发请求上限和请求超时，配合已有请求体大小限制，降低单客户端或突发流量压垮服务的风险。
- **Python 宽泛异常收敛**：`python/src/train/app/data.py` 的 CSV fallback 已改为捕获 `pd.errors.ParserError` / `ValueError` 并记录日志；`manifest.py` 的 git commit 获取已改为捕获 `subprocess.CalledProcessError` / `OSError`。
- **FeatureHash 零 vocab 防护**：`FeatureHash::new()` / `with_scope()` 已返回 `Result` 并拒绝 `vocab_size == 0`，不再使用生产路径 `assert!()`。
- **unsafe 注释**：插件动态库加载、FFI 调用和 safetensors mmap 的 unsafe 块已补充 `// SAFETY:` 注释。
- **Python 列式训练 batch**：`stream_file_batches()` 已避免 pandas chunk 到 `to_dict("records")` 的往返转换，直接 yield feature/label 列式 batch。
- **周期 checkpoint 和 resume**：训练侧已经保存 optimizer、loss、EMA、step、epoch、best score、stale epochs、周期 checkpoint 计数器和 Python/NumPy/Torch/CUDA RNG 状态，支持 `resume_from_checkpoint()`。
- **发布产物结构化**：`TrainingArtifactManager` 已统一 run/checkpoints/serving/configs 目录，默认复制 feature/model config，并写出 run manifest 与 serving manifest。
- **权重加载校验**：Rust registry 在 `VarMap::load()` 前检查 safetensors key 缺失和 shape mismatch，是重要稳定性防线。
- **FeatureDag 深化拆分**：Rust/Python 均已拆出 `DagBuilder`、`DagExecutor`、`FeatureInfo`，Python 额外拆出 `DagPreprocessor` / `TrainingPreprocessor`；训练、评估、metrics、quality 和 Rust serving/demo 调用方已迁移到新 seam，`FeatureDag` 仅作为兼容 facade。
- **operator type 枚举化**：Rust `OperatorDef.op_type` 已从 `String` 改为 `OpType` enum，Python 已从 `str` 改为 `OpType(str, Enum)`；schema 推导、quality padding 识别和 Rust registry 不再用字符串判断 operator 类型。
- **apply_relation 去重**：ESMM/GDCN-ESMM 的 relation 应用逻辑已收敛到 `layers::towers::apply_relation()`，避免多任务概率关系重复实现。
- **任务契约 seam**：Python 已新增 `TaskContract` / `TaskSpec`，集中生成 `task_names`、`label_col_map`、`output_kinds` 和 manifest `task_specs`，训练、导出和 serving 查询不再各自解析任务语义。
- **模型 serving 元数据扩展**：manifest 与 `/models` / version serving info 已暴露 `loaded_at`、schema/config hash、tasks、task_specs、label map、metrics 和 weight binding 等关键元数据。
- **模型权重绑定检查**：已新增 Python `check_weight_bindings.py` 和 Rust `validate_manifest` 二进制，用于从示例模型导出 safetensors/manifest 并触发 Rust key/shape 校验。
- **原生多任务输出契约阶段 1-4**：Rust/Python 已共享 `output_contract.version: 1`
  schema 与 fixtures；双端 `OutputHead` 执行 typed relation DAG；Python
  `ObjectiveEngine` 计算契约损失；全部 8 个模型已输出公开 projection，并保留内部训练
  节点。MMoE 支持多命名 backbone representation。
- **Rust public API 文档与 warning 清零**：Rust public API 已补齐 rustdoc，`#![warn(missing_docs)]` 已启用；`cargo check` 和 `cargo doc --no-deps` 当前无 warning。

## 3. 架构分析

### 3.1 FeatureDag 与特征契约

当前特征系统的 Module seam 已从单一 `FeatureDag` 深化为构建、执行、元数据和预处理四个角色：

- `FlowConfig` 描述 source、operator、embed、data source 和 role。
- `DagBuilder` 完成拓扑排序、算子创建、schema 推导、验证和预编译 `ExecutionPlan`。
- `DagExecutor` 负责执行 seam，Rust serving 和 demo inference 使用 plan-based execution，Python feature quality 已迁移到 executor + metadata seam。
- `FeatureInfo` 负责 embeddable features、source kind、node defs/source defs 等只读元数据查询，模型构建和 broadcast 策略不再依赖 `FeatureDag` facade。
- `DagPreprocessor` / `TrainingPreprocessor` 负责训练侧从列式 batch 或兼容 list-of-dicts 输入到模型 tensor 的转换。
- 当前已支持 `ParsedFeatureHash` 和 `ConcatHash` 两个融合算子，减少“解析 + hash”链路中的中间对象和配置层级。

优势：

- YAML 是训练和推理共享的特征契约，降低特征漂移风险。
- DAG 构建阶段已有 source 消费率、输出利用率、data source 引用和拓扑校验。
- Rust 推理主路径已收敛到预编译 plan；Rust broadcast 模式可预计算 user-only 子图并复用到 item batch。
- `FeatureSchema` 已开始承载 dtype、list 长度、pooling、default、role 等运行时契约信息。
- `OpType` 已双端枚举化，schema 推导、quality padding 识别和 registry 创建路径不再依赖 operator type 字符串判断。

剩余风险：

- 默认值解析虽然已集中到 `src/feats/defaults.rs` 的 `source_default()`，但 Python `_parse_default()` 仍需持续保持语义一致。
- Python 训练预处理仍通过 `TrainingPreprocessor` 包装兼容 facade，后续可继续迁移到直接组合 `DagExecutor` + `DagPreprocessor`。
- `FeatureSchema` 还没有完全成为唯一执行契约，部分算子参数仍在 DAG 构建时用 `unwrap_or(...)` 默认值解释。
- Python/Rust 已对 operator params 做 allowlist、required 和基础类型校验，但还不是每个算子独立 typed config；复杂参数的语义校验仍主要在算子工厂中完成。

改进方向：

- 继续压缩 `FeatureDag` facade 的使用面，最终只保留兼容入口或彻底删除 facade。
- 为 operator params 建立 typed config，而不是在各算子工厂中散落读取和默认值。
- 继续把复杂 operator params 收敛为 typed config，而不是只依赖通用 YAML allowlist。
- 让 `FeatureSchema` 成为 tensor 构造、默认值、seq_len、pooling 和 feature contract 的唯一来源。
- 对所有融合算子的 parse mode、seq_len、padding、hash scope 增加 Python/Rust golden consistency 覆盖。

### 3.2 模型与权重加载

模型层已从中央 enum 逐步转为 registry 模式：

- Rust `src/models/mod.rs` 的 `ModelConfig` 是 `type` + flattened YAML params，`REGISTRY` 注册 `lr`、`deepfm`、`mmoe`、`esmm`、`gdcn_esmm`、`unimixer`、`token_mixer_large`、`rankmixer`。
- Python `python/src/train/models/__init__.py` 使用 `register_model()` / `build_model()` 注册同名模型。
- UniMixer、TokenMixer-Large 和 RankMixer 通过外部 `FeatureTokenizer` 注入，避免模型内部重复构建特征规格。
- Rust registry 已支持 manifest-driven loading、多版本、默认版本、alias、routing、feature contract 查询和权重 key/shape 校验。

优势：

- 新增模型不需要修改中央 enum，但仍需要双端注册和双端实现。
- 模型配置不重复声明特征，减少配置耦合。
- safetensors key/shape 校验能提前暴露权重不兼容。
- 全部模型均可使用显式 `output_contract`，并保持
  graph/objective/metric/output 四部分边界。

剩余风险：

- Rust 和 Python 模型仍是双份实现，新增 layer、task tower 或 naming prefix 时必须人工保持 `state_dict` key 对齐。
- Rust/Python 已对模型 params 做 allowlist、required 和基础类型校验，但模型参数仍不是独立 typed config，错误信息和默认值语义还可以继续集中化。
- serving manifest 尚未保存规范化 `output_contract` 或摘要，服务端元数据仍主要暴露
  legacy tasks/label map/metrics。

改进方向：

- 将 `check_weight_bindings.py` / `validate_manifest` 纳入 CI，覆盖所有示例模型的 state_dict key 和 shape 对齐。
- 为 Rust/Python 模型参数建立更细的 typed config，减少通用 YAML 参数读取和默认值分散。
- 将规范化 `output_contract` 及摘要写入 manifest，并让 serving 元数据明确公开输出。
- 完成配置迁移后再提供 legacy 机械适配器和删除旧执行路径。

### 3.3 训练产物与发布

训练侧已有 `TrainingArtifactManager`，发布链路已经从“只有权重文件”演进为结构化产物：

- run 目录保存 checkpoints、latest/best alias、resume state、run manifest 和 serving 目录。
- serving 目录包含 `model.safetensors`、`model.manifest.yaml` 和 `configs/` 下的 feature/model config 副本。
- manifest 记录 weights、feature config、model config 的路径与 sha256，以及 tasks、label map、metrics、weight binding。
- Rust registry 加载 manifest 时会校验 feature/model/weight sha256 和 schema version。

剩余风险：

- alias 和 routing 当前通过 HTTP 修改后只存在于 `ModelRegistry` 内存中，重启丢失，也无法审计。
- 默认版本选择仍主要来自加载顺序/版本字符串逻辑，缺少发布索引或显式控制面。
- 如果显式 `--publish-path` 指到 run 目录外，仍需确认 serving manifest 中引用的配置副本始终自包含。
- artifact 目录下可能存在 demo safetensors 和临时训练产物，若被纳入版本控制会造成仓库膨胀或模型泄漏风险，需要用 `git status`/`.gitignore` 持续约束。

改进方向：

- 将 alias/routing/default version 持久化到发布元数据或独立版本索引文件。
- 服务启动时只依赖 serving 归档内的配置和权重，避免隐式依赖仓库 `examples/`。
- 建立发布索引：记录 active/default/canary/rollback 目标和变更时间。
- 在 manifest 中记录 feature quality summary，并让服务端可查询加载后的质量摘要。

### 3.4 错误处理与鲁棒性

生产 server 启动和预测路径已经比上一版稳健，但错误处理仍不完整：

- `src/main.rs` 是 demo binary，仍有 `expect()`、`unwrap()` 和 unsupported feature type `panic!()`，应与生产 server 风险分开看待。
- `src/bin/bench.rs` 是压测工具，仍有较多 `expect()` / `panic!()`，对生产服务影响较低，但会影响压测输入错误的可诊断性。
- 模型构建中存在对非空 hidden dims 的隐含假设，例如部分模型取 `last().unwrap()`；如果配置缺少关键 dims，错误信息不够面向用户。
- registry 部分读锁 poisoned 时使用 `.ok()?` 返回 `None`，表现可能接近“模型不存在”，不利于定位内部状态损坏。
- API error response 会把内部错误 message 原样返回客户端，可能泄露路径、权重 key 或配置细节。

改进方向：

- 把 demo/bench 的 panic 与 server 生产路径分级治理，避免把非生产工具误判为线上 P0。
- 模型配置校验前置化：缺少必填 dims、tower、task_config 时返回带字段路径的配置错误。
- registry poisoned lock 应统一记录 `error!` 并返回内部错误，而不是静默退化为 not found。
- API error 对外返回稳定 code 和简短 message，详细内部错误只写日志或 trace。

### 3.5 安全与防御

已完成：

- CORS 已改为白名单模式。
- 请求体大小已有上限。
- 已有全局 rate limit、并发限制和请求超时。
- 已知 unsafe 块已有安全说明。

仍然缺失：

- **认证缺失**：所有 HTTP 端点无 API key、mTLS 或其他认证；alias/routing 修改端点也暴露在同一服务上。
- **按客户端/租户限流缺失**：当前限流是服务级全局保护，还不能按 IP、租户或 API key 隔离额度。
- **管理面与预测面未隔离**：模型查询、alias/routing 修改、预测请求在同一个 Router 中，没有权限分级。
- **插件路径未约束**：`PluginOp` 仍可加载配置中给出的本地动态库路径，没有白名单、签名校验或禁用开关。
- **Docker 仍以 root 运行**：`docker/Dockerfile` 和 `docker/Dockerfile.mkl` 没有 `USER` 指令。

改进方向：

- 增加 API key 或 mTLS，至少保护 alias/routing 等管理端点。
- 在全局保护基础上增加按 IP、租户或 API key 的细粒度限流策略。
- 将管理端点拆到独立 Router、端口或部署单元，明确权限 seam。
- 为插件增加 allowlist、禁用开关和路径 canonicalize 校验；生产默认禁用动态插件。
- Docker runtime 添加非 root 用户，并明确只读 root filesystem、模型目录挂载权限。

### 3.6 工程基础设施

当前工程治理仍是主要短板：

- 无 `.github/workflows/`、`.gitlab-ci.yml` 或 Jenkinsfile，测试、lint、构建和安全扫描依赖人工执行。
- `Cargo.lock` 已入仓，Dockerfile 使用 `cargo build --release --locked`，但缺少 CI 防止锁文件约束回退。
- 无 `rustfmt.toml`、`clippy.toml`，Rust 风格和 lint 主要依赖默认规则。
- Python ruff 配置仍较弱，仅忽略 `E402`，没有启用 `I`、`B`、`UP`、`SIM` 等规则集。
- Mypy 对大量核心模块设置 `ignore_errors = true`，包括 config、DAG、models、trainer、metrics 等，类型检查实际覆盖有限。
- 无 `.pre-commit-config.yaml`。
- 依赖安全扫描缺失：无 `cargo-audit`、`cargo-deny`、Dependabot/Renovate 配置。

改进方向：

- 增加 CI：`cargo fmt --check`、`cargo clippy --locked`、`cargo test --locked`、`uvx ruff check`、`uvx ruff format --check`、`uv run mypy`、`uv run pytest`。
- 增加 `cargo-audit` / `cargo-deny` 和 Python 依赖更新机器人。
- 渐进启用 ruff `I`、`B`、`UP`、`SIM`，逐步移除 mypy `ignore_errors`。
- 添加 `.editorconfig` 和 `.pre-commit-config.yaml`，把格式化和基础 lint 前移到提交前。

### 3.7 代码重复与浅 Module

当前最需要“加深”的 Module 不是简单增加新接口，而是让高变化复杂度集中到更少、更深的 seam 后面：

- ~~`FeatureDag` 同时承担 DAG 构建、schema 推导、验证、单样本执行、批量执行、tensor 预处理和调试支持，Interface 接近 Implementation 复杂度。~~ ✅ 已拆为 `DagBuilder` / `DagExecutor` / `FeatureInfo` / `DagPreprocessor`，`FeatureDag` 保留兼容 facade。
- `Trainer` 同时承担训练循环、数据迭代、评估、EMA、checkpoint、resume、artifact、日志和 prefetch，Locality 较弱。
- ~~Rust operator params 解析散落在 `FeatureDag` 内部~~ ✅ 已重构为 registry 模式，每个算子自包含 `create()` 工厂函数
- ~~operator type 使用字符串判断~~ ✅ 已重构为 Rust/Python 双端 `OpType` 枚举，schema、quality 和 registry 路径使用枚举分发。
- Python/Rust operator 双端实现仍依赖人工同步，缺少一个统一的 operator contract 测试矩阵。

改进方向：

- 继续收敛 `FeatureDag` facade，减少兼容层暴露的旧接口，最终让新代码只依赖 `DagBuilder`、`DagExecutor`、`FeatureInfo` 和 `DagPreprocessor`。
- `Trainer` 长期拆出 `CheckpointManager`、`ResumeState`、`TrainingLoop`、`EvaluatorAdapter`，优先让 checkpoint/resume 成为深 Module。
- 为 operator params 建立 typed config 和 shared golden fixtures，让新增算子的测试面成为稳定 Interface。
- 删除重复常量和胶水逻辑，例如训练入口的 null marker、batch 配置和 label mapping 处理。

## 4. 功能分析

### 4.1 已具备能力

当前项目已经具备以下核心能力：

- 共享 YAML 特征编排，覆盖 Bucketing、DictMapper、StringParser、JsonExtractList、
  ListStringParser、Split、FlatSplit、ExpressionOp、CrossFeature、ListOverlap、SequenceOp、
  StringConcat、FeatureHash、Log1p、PluginOp、ParsedFeatureHash、ConcatHash 等 17 个算子。
- Python 单文件训练、discover 训练、多模型训练入口，共享列式 batch 预处理和可选 prefetch。
- 多任务 loss、评估、early stopping、EMA、周期 checkpoint、epoch-end checkpoint、resume 和 manifest。
- 任务级配置落到模型 YAML；legacy 模型由 `TaskContract/MultiTaskLoss` 执行，原生
  ESMM 由 `OutputContract/ObjectiveEngine` 执行。
- Rust HTTP 推理支持 pointwise `/predict` 和 broadcast `/predict/broadcast`。
- Rust 推理支持 plan DAG、broadcast user 子图预计算、多模型热加载、多版本、alias 和 routing。
- Feature contract 查询接口可暴露每个模型/版本需要的输入字段、dtype、default 和 data source；模型查询接口可暴露 schema hash、tasks、metrics、label map、加载时间和 weight binding。
- Golden consistency 测试覆盖 Python/Rust 特征处理一致性。
- safetensors 权重 key 与 shape 校验。
- 示例模型权重绑定检查脚本覆盖 Python 导出到 Rust manifest 加载路径。
- benchmark 工具支持 synthetic 和真实 discover 输入压测。
- `docs/notes/http_benchmark_report.md` 已记录 GDCN+ESMM 和 UniMixer 在 broadcast 场景下的端到端压测结果。

这些能力说明项目已经具备工程化迭代基础。下一阶段重点不是堆更多模型，而是收紧配置契约、补齐安全控制面、建立 CI 质量闸门，并把性能优化建立在稳定基线之上。

### 4.2 功能缺口

需要优先补齐的功能缺口：

- 认证和权限：预测端点、模型查询端点、alias/routing 管理端点目前都无认证。
- 限流和过载保护：全局 rate limit、并发限制和请求超时已具备，但仍缺少按 IP、租户或 API key 的额度隔离和更细粒度 backpressure。
- 发布控制面持久化：alias、routing、default version 目前是内存状态，缺少可审计、可回滚的持久化发布索引。
- 模型级 schema 元数据：`/models` 已返回 schema hash、tasks、metrics、label map、weight binding 和加载时间，但 embeddable schema、feature quality summary 和更细的兼容状态仍未暴露。
- 配置兼容策略：feature config、model config、manifest schema 的版本兼容规则尚未系统化。
- 模型/operator 参数严格校验：Rust/Python 对未知字段、错误类型、缺失必填参数的处理仍不一致。
- 线上可观测性：已有 parse/dag/tensor/forward/response 耗时，但缺少 default hit rate、空序列比例、截断次数、FeatureHash cache、broadcast 子图 skip 数等指标。
- 数据质量闭环：验证集 feature quality summary 已进入 manifest；完整训练流的 embedding bucket report 已生成并由 manifest 引用，但服务端查询和线上日志链路仍未暴露。
- CI/CD 自动化：缺少自动化质量闸门。
- 类型检查：mypy 对核心模块仍大面积 `ignore_errors = true`。

## 5. 性能分析

### 5.1 Rust 推理链路

当前 Rust 推理链路大致为：

1. JSON rows 转 `FeatureRow` / typed columns。
2. `ExecutionPlan::execute_plan()` 执行 DAG。
3. embeddable feature 列转 Candle tensor。
4. 模型 forward。
5. tensor 输出 flatten 后转 JSON response。

已完成或已有优化：

- DAG 使用预解析列 id 的 plan 执行。
- broadcast 模式预计算 user-only 子图，再对 item candidates 复用。
- `RequestTimer` 已记录 parse、dag、tensor、forward、response 和 batch size。
- `feature_column_to_tensor()` 按配置的 `seq_len` 做 padding/truncation。
- `ParsedFeatureHash` / `ConcatHash` 减少解析和 hash 之间的中间分配。
- GDCN cross/gate GEMM 和 UniMixer token/block 小矩阵乘路径已有针对性优化。
- `FeatureHash` 已有 hit/miss/size 计数。

性能风险：

- JSON row-wise 请求结构和 `HashMap<String, Value>` 对高 QPS 场景仍有明显开销。
- DAG 到 tensor 构造之间仍存在 clone 和按行 flatten 的中间对象。
- `FeatureHash` scalar batch 路径持有全局 write lock 处理整批数据，高并发下可能成为热点。
- `FeatureHash` cache 无容量上限，高基数字段会造成内存持续增长。
- response 阶段将每个 tensor flatten 到 `Vec<f32>`，任务数和 batch size 增大时会增加拷贝成本。
- UniMixer 对 CPU 后端仍敏感，macOS Accelerate/native 和 Linux native/MKL 需要持续分别压测。

改进方向：

- 固化压测矩阵：pointwise/broadcast、不同 batch size、不同模型、不同 CPU 后端。
- 对 `FeatureHash` 增加容量限制、分片缓存或可关闭配置，并把 hit/miss/size 暴露到服务指标。
- 增加列式请求 API 或二进制协议作为高吞吐可选路径。
- 优先优化 int scalar 和 fixed sequence tensor 构造，减少 clone 和临时 Vec。
- 在 broadcast 日志/指标中记录 user op 数、item/cross op 数、skip op 数和 precompute 耗时。

### 5.2 Python 训练链路

当前 Python 训练链路已经从 row records 向 columnar batch 前进：

- `stream_file_batches()` 使用 pandas chunk 读取，只保留 feature/label `usecols`，再 yield `dict[str, list]`。
- `TrainingPreprocessor` 通过兼容 facade 优先处理列式输入，list-of-dicts 作为兼容路径。
- `Trainer` 支持 `ThreadPoolExecutor` prefetch，能隐藏一部分数据读取和预处理开销。
- 训练开始会记录数据摘要、reader 配置、prefetch、checkpoint interval、optimizer 和任务映射，便于定位无监督 batch 或标签列不匹配。

性能风险：

- 底层仍是 pandas chunked CSV，超大训练集会受 CSV parse、dtype conversion 和 Python list materialization 限制。
- `build_item_index()` 仍构造大字典，物品规模很大时内存压力明显。
- Python 预处理链路仍包含部分逐行 DAG 语义和 Python list/tensor 转换，算子复杂时 CPU 开销明显。
- prefetch 只是隐藏部分 CPU 开销，还不是读取、预处理、设备拷贝的完整 producer/consumer 流水线。

改进方向：

- 大数据训练引入 Polars LazyFrame 或 Arrow RecordBatch，减少 CSV 到 Python object 的转换。
- 让更多 operator 支持真正的列式 batch 实现，而不是列式输入后内部仍按行处理。
- 拆分读取、预处理和 device transfer，形成有界队列流水线。
- 对关键算子补充 batch consistency 测试：单样本执行与 batch 执行输出必须一致。

## 6. 改进优先级

### P0：生产安全底线与质量闸门

1. **添加认证和权限控制**：至少用 API key 或 mTLS 保护 HTTP 服务；alias/routing 管理端点必须先受保护。
2. **建立 CI/CD 基础管线**：强制 `cargo fmt --check`、`cargo clippy --locked`、`cargo test --locked`、`uvx ruff check`、`uvx ruff format --check`、`uv run pytest`。
3. **把 alias/routing/default version 持久化**：避免服务重启丢失发布控制状态，并支持审计和回滚。
4. **收紧模型/operator params 校验**：operator type 已枚举化，但 params 仍是松散 YAML；未知字段、错误类型、缺失必填参数必须 fail fast，不能静默走默认值。
5. **Docker 非 root 运行**：`docker/Dockerfile` 和 `docker/Dockerfile.mkl` 添加非 root `USER`，并确认模型目录权限。
6. **API error 脱敏**：对外只返回稳定 code 和必要 message，内部路径和详细错误写入日志。
7. **细粒度限流**：在现有全局 rate limit/concurrency/timeout 基础上，增加按 IP、租户或 API key 的额度隔离。
8. **CI 纳入权重绑定检查**：将 `scale_rec_demo.check_weight_bindings` 和 Rust `validate_manifest` 纳入质量闸门，防止 Python/Rust 模型命名漂移。

### P1：可观测性与发布治理

1. 增加模型发布索引文件，记录 default/canary/rollback/alias/routing 变更。
2. 增加 Prometheus/OpenTelemetry 指标导出，至少覆盖请求量、错误量、延迟分段、batch size、broadcast item count。
3. 增加 feature 质量和默认值命中指标：default hit rate、empty sequence、truncation、FeatureHash cache hit/miss/size。
4. ~~将完整训练流 embedding bucket report 写入发布产物并由 manifest 引用。~~ ✅ 已完成；下一步在 `/models` 或 feature contract 查询中暴露摘要。
5. 插件机制增加 allowlist、禁用开关和路径 canonicalize 校验；生产默认禁用动态插件。
6. 增加 `cargo-audit` / `cargo-deny`、Dependabot/Renovate 和 `.pre-commit-config.yaml`。
7. 渐进启用 ruff `I`、`B`、`UP`、`SIM`，逐步移除 mypy `ignore_errors`。

### P2：性能基线与吞吐优化

1. 建立固定压测矩阵，把 pointwise/broadcast、batch size、模型、后端、P50/P95/P99/P99.9、RPS 写入 docs 或 CI artifact。
2. 为 `FeatureHash` 增加 cache 容量限制、分片或可关闭配置。
3. 优化 tensor 构造阶段，减少高频 int scalar 和 fixed sequence 的 clone/flatten。
4. 增加列式请求 API 或二进制协议，降低 JSON row-wise 解析成本。
5. Python 训练引入 Arrow/Polars-first pipeline，减少 pandas/list/object 转换。
6. 将 prefetch 演进为读取、预处理、device transfer 的有界 producer/consumer 流水线。

### P3：重构与平台化

1. 继续收敛 `FeatureDag` facade：删除新代码对旧 facade 属性/方法的依赖，最终只保留兼容入口或移除。
2. 拆分 `Trainer`：`CheckpointManager` + `ResumeState` + `TrainingLoop` + `EvaluatorAdapter`。
3. ~~为所有公共 API 补充 rustdoc/docstring，并在 CI 中逐步启用 missing docs 检查。~~ ✅ 已完成，`#![warn(missing_docs)]` 已启用。
4. ~~标准化 operator 注册和 operator type 分发，降低新增算子的 Rust/Python 双端维护成本。~~ ✅ 已完成，双端均使用 registry 模式和 `OpType` 枚举。
5. ~~为模型 state_dict key 对齐建立自动化测试或导出检查脚本。~~ ✅ 已新增 `check_weight_bindings.py` 和 `validate_manifest`，后续需纳入 CI。
6. 建立模型发布、回滚、灰度和兼容检查流程，把 runtime alias/routing 接入持久化控制面。
7. 将 manifest 已引用的 embedding bucket report 和 feature quality 摘要在服务端加载后可查询。

## 7. 推荐执行路线

**第 1 阶段（1-2 周）**：认证、CI、Docker 非 root、API error 脱敏、细粒度限流，并把权重绑定检查纳入 CI。目标是补齐生产安全底线和质量闸门。

**第 2 阶段（2-4 周）**：alias/routing 持久化、feature quality manifest/查询、配置 strict validation、插件 allowlist。目标是让发布控制、数据质量和配置错误可审计、可回滚、可诊断。

**第 3 阶段（1-2 月）**：固定压测矩阵、FeatureHash cache 治理、tensor 构造优化、Python Arrow/Polars pipeline 试点。目标是在可复现基线上提升吞吐。

**第 4 阶段（2-4 月）**：继续收敛 `FeatureDag` facade、拆分 `Trainer`、标准化 operator params，建立发布兼容检查流程。目标是提升长期演进的 Locality 和 Leverage。

## 8. 验收建议

每一轮改进至少满足以下验收条件：

- Rust：`cargo fmt --check`、`cargo clippy --locked`、`cargo test --locked`、`cargo check --locked` 通过。
- Python：`PYTHONPATH=python/src uv run pytest python/tests/ -v`、`uvx ruff check python/src/`、`uvx ruff format --check python/src/` 通过。
- 类型：涉及 Python 核心训练模块时，应明确 mypy 覆盖变化，不能继续扩大 `ignore_errors`。
- CI：上述检查在 CI 中自动执行，PR 合并前必须通过。
- 一致性：涉及特征、算子、模型结构或权重命名时，必须跑 Golden consistency 或 `scale_rec_demo.verify_all` 对应路径。
- 发布：涉及 manifest、alias、routing、default version 时，必须验证服务重启后状态仍可恢复。
- 性能：涉及推理或训练性能时，必须记录优化前后的模型、后端、batch size、P50/P95/P99/P99.9、RPS 或 per-batch timing。
- 安全：CORS 必须为白名单；认证、限流、请求体大小限制和管理端点权限必须存在。
- 文档：新增公开 API、配置字段或发布 manifest 字段必须同步更新 docs 和示例 YAML。

总体建议是：当前代码库已经完成多项早期 P0 稳定性修复，下一步应把重心从“消除显性 panic/println”转到“安全边界、发布控制、配置严格性、CI 闸门和可观测性”。性能优化应基于固定压测矩阵进行，重构应围绕更深的 Module seam 渐进推进，避免大范围重写。
