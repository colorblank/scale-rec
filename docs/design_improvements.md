# 设计改进建议

本文从推荐系统算法工程师和系统架构师两个视角，梳理当前 scale-rec 代码库后续值得改进的设计点。重点不是单点 bug，而是训练、推理、特征、模型发布和线上服务长期演进中的系统性风险。

## 推荐系统算法工程视角

### 1. 强化特征 schema 和类型推导

当前 YAML 已经统一描述 `sources`、`operators` 和 `embed`，但配置仍偏弱类型。DAG 构建阶段应进一步推导并校验每个特征的：

- `dtype`：`int`、`float`、`string`、`list[int]`、`list[float]`、`list[string]`
- `rank/shape`：标量、变长序列、定长序列
- `nullable/default`：缺失值是否允许、默认值类型是否匹配
- `cardinality`：词表大小、hash bucket 数、低基数字段枚举范围
- `pooling`：算子输出是否支持配置的 `first/mean/sum/max/flatten`

建议引入 `FeatureSchema` 或类似结构，由 DAG 编译阶段从 source 和 operator 逐层推导。配置错误应在训练或模型上线前 fail fast，而不是在运行时默默落默认值。

### 2. 建立 Python/Rust golden consistency 测试

项目目标是 Python 训练、Rust 推理共享 YAML 和模型结构。当前双端分别实现算子和模型，长期容易发生语义漂移。建议建立固定 golden fixtures：

- 同一份 YAML
- 同一批 raw rows
- Python DAG 输出和 Rust DAG 输出逐字段对齐
- 同一 safetensors 权重、同一 batch 输入，Python/Rust logits 在误差阈值内对齐

新增算子、修改 pooling、修改模型命名或权重路径时，都应更新或运行这类一致性测试。

### 3. 增加特征质量和漂移监控

推荐系统线上效果高度依赖数据质量。建议训练侧和推理侧都记录以下指标：

- 每个 source 的缺失率、默认值命中率
- 每个 embeddable feature 的空序列率、序列长度分布
- hash bucket 使用率、top bucket、碰撞风险指标
- 数值特征分位数、均值、方差、异常值比例
- 训练集和线上请求之间的 PSI/KL/分布漂移

这些指标应进入训练报告、模型 manifest 和线上监控，便于发现“模型没变但特征坏了”的问题。

### 4. 抽象统一的任务和 loss 定义

当前多任务模型、label map、loss 和指标逻辑分散在模型配置、训练入口和 metrics 中。建议引入显式 `TaskSpec`：

```yaml
tasks:
  - name: click
    label: is_click
    loss: bce
    weight: 1.0
  - name: stay
    label: stay_time
    loss: weighted_bce_stay
    mask: stay_time >= 0
    weight: 0.2
```

`TaskSpec` 应统一驱动：

- 模型输出 task 名称
- label column 映射
- loss 类型
- mask 条件
- sample weight
- AUC/logloss/自定义指标
- ESMM/多任务概率关系

这样可以避免不同训练脚本对同一个任务使用不同 label 或 loss。

### 5. 补齐样本权重、负采样和偏差修正能力

当前训练流程更接近 demo pipeline。生产推荐训练通常还需要：

- 曝光位置偏差修正
- 负采样策略和采样权重回填
- 时间衰减权重
- 用户、物品、场景分桶采样
- 延迟转化窗口和 label 成熟期
- 任务级动态权重或不确定性权重

建议 batch 数据结构支持 `sample_weight`、`task_mask`、`context` 等字段，不要只传 features 和 labels。

### 6. 完善 FeatureHash 的治理能力

默认使用 FeatureHash 可以降低词表维护成本，但也会带来可解释性和碰撞问题。建议增加：

- hash namespace / salt / version
- hash bucket 分布报告
- 重要字段的冲突率估算
- top raw value 到 bucket 的采样反查
- 低基数字段优先 DictMapper 的配置检查

对于核心字段，训练报告应展示 hash 后分布，避免大字段被少量 bucket 异常支配。

### 7. 模型导出需要 manifest

当前权重、feature config、model config 仍是松散文件。建议每次训练导出一个 `model_manifest.yaml` 或 `model_manifest.json`：

```yaml
model_id: discover_esmm
model_version: 20260525_120000
code_commit: abc1234
feature_config_sha256: ...
model_config_sha256: ...
weights_file: model.safetensors
tasks:
  - click
  - cvr
metrics:
  click_auc: 0.74
  cvr_auc: 0.68
```

Rust server 应加载 manifest，而不是靠文件命名猜测模型配置。这样能避免权重、模型配置和特征配置错配。

## 系统架构视角

### 1. 显式化 Rust/Python 双端契约

共享 YAML 只是第一步，还需要明确契约：

- YAML schema 版本
- 算子输入输出规范
- pooling 和 sequence 语义
- safetensors key 命名规范
- 不兼容变更策略
- golden fixture 测试要求

新增算子或模型时，应该同时提交 Python 实现、Rust 实现、契约文档和一致性测试。

### 2. 收敛 DAG 执行路径

Rust 当前存在单样本执行、batch 执行和 plan 执行等路径。路径越多，语义漂移和回归风险越高。建议以编译后的 `ExecutionPlan` 为核心：

- single-row execute 作为 plan execute 的薄封装
- batch execute 作为主路径
- broadcast 复用同一套列式执行逻辑
- debug/tracing 作为 plan 执行的可选 hook

这样可以减少同一业务逻辑在多处重复实现。

### 3. 模型加载和发布机制需要生产化

`ModelRegistry` 应从 demo 风格的目录扫描升级为版本化模型注册表：

- 按 manifest 加载模型
- 支持 model version 和 stage：`candidate`、`online`、`rollback`
- 加载前做配置校验和权重 key 校验
- 加载后自动 smoke predict
- 线上切换使用原子替换
- 保留上一版本用于快速回滚

服务 API 也应支持查询当前模型版本、加载时间、配置 hash 和健康状态。

### 4. 错误处理应面向线上服务

在线推理路径应避免 `unwrap()` 和宽松吞错。建议区分错误类型：

- `BadRequest`：请求 JSON 格式错误、缺少必要字段、类型不匹配
- `FeatureError`：DAG 执行错误、算子参数错误
- `ModelError`：tensor shape 不匹配、模型 forward 失败
- `RegistryError`：模型不存在、模型未加载、权重加载失败
- `InternalError`：非预期错误

错误响应应包含稳定错误码、可读 message 和 request id，便于线上排障。

### 5. 增强可观测性

推荐推理服务至少应暴露以下指标：

- QPS、错误率、延迟分位数
- DAG 耗时、模型 forward 耗时、总耗时
- batch size 分布
- 每个模型版本请求量
- 默认值命中率、缺字段率
- 每个算子失败次数
- `/predict` 和 `/predict/broadcast` 分开统计

日志应使用结构化字段，包括 `model_id`、`model_version`、`request_id`、`batch_size`、`latency_ms` 和错误码。

### 6. 配置校验分级

当前 orphan source/output 之类问题容易被 warning 掩盖。建议引入配置校验等级：

- `error`：未知输入、环、重复输出、embed vocab 非法、pooling 与输出类型不匹配
- `warning`：orphan source、orphan output、高默认值命中风险
- `info`：source/operator/embeddable feature 数量统计

训练和上线默认 strict 模式；本地实验可以放宽部分 warning。

### 7. 稳定 Python 工程环境

Python 训练环境需要更容易复现。建议：

- 明确支持 Python 版本，例如 3.11 或 3.12
- 固定 PyTorch 主版本和 CUDA/CPU 组合
- CI 中运行 `ruff check`、`ruff format --check`、`pytest`
- 提供最小训练 smoke：生成小数据、训练 1 epoch、导出 safetensors
- 提供 Python/Rust 推理对齐 smoke

训练环境稳定性比使用最新语言版本更重要。

### 8. 补齐端到端测试矩阵

建议按风险从高到低补测试：

- feature config discover 能成功 build DAG
- 同一 batch Python/Rust DAG 输出一致
- 训练小模型并导出 safetensors
- Rust 加载 safetensors 并完成 predict
- server `/predict` 和 `/predict/broadcast` API 测试
- 模型 manifest 校验失败用例
- 缺字段、空序列、异常类型、未知 item 的边界用例

## 建议优先级

### P0：先保证训练/推理契约正确

- Python/Rust golden consistency 测试
- FeatureSchema 类型和 shape 校验
- 模型 manifest
- server 加载前权重 key 校验

### P1：提升线上可靠性和排障能力

- 结构化错误码
- Prometheus 指标
- 默认值命中率和特征缺失率监控
- 模型版本化加载和原子切换

### P2：提升算法迭代质量

- TaskSpec 统一任务、label、loss、metric
- 样本权重和 task mask
- hash 分布报告
- 特征漂移报告

### P3：再做性能和模型结构优化

- DAG 执行路径收敛
- 更细粒度 batch/plan 优化
- UniMixer 或多任务结构优化
- 更复杂的去偏和校准策略

当前最高收益不是继续堆复杂模型，而是把特征契约、训练推理一致性、模型发布和数据质量闭环做扎实。
