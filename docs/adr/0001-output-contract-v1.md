# 使用显式输出契约统一多任务模型语义

状态：已接受，阶段 1-3 已落地。

## 背景

旧模型把任务塔、ESMM 概率关系、公开输出和损失规则分散在模型实现、训练代码及
serving 适配逻辑中。相同任务名在不同模型里可能代表 logit 或概率，导致损失选择、
指标计算和 Rust/Python 推理一致性依赖隐式约定。

## 决策

新增版本化的 `output_contract.version: 1`，统一描述以下五部分：

- `graph.towers`：从 backbone 的命名表示构建标量任务塔。
- `graph.relations`：在塔输出之上构建无参数、有类型的关系 DAG。
- `objectives`：声明训练目标、标签、损失、权重和可选样本 mask。
- `metrics`：独立声明评估节点、标签、指标和可选样本 mask。
- `outputs`：将内部节点投影为稳定的公开输出名称。

塔只允许输出 `binary_logit`、`regression` 或 `score`。概率必须通过显式
`sigmoid` 关系产生，`multiply` 只接受两个及以上概率输入，`add` 只接受两个及以上
回归输入，`identity` 保留输入类型。关系按 DAG 拓扑执行，循环、未知引用、重复名称和
未消费节点均在构建前拒绝。

损失和节点类型严格匹配：

- `binary_cross_entropy_with_logits`、`weighted_bce_stay` 接受 `binary_logit`。
- `binary_cross_entropy` 接受 `probability`，并使用显式 epsilon 截断。
- `mse`、`mae`、`huber` 接受 `regression` 或 `score`。

指标按节点类型处理分类值。`auc` 接受 logit 或概率；logit 会在指标入口转换为概率，
概率不会重复执行 sigmoid。`logloss` 只接受概率，
`mae/mse` 只接受回归值或排序分数。serving 只序列化 `outputs` 指定的节点，不再自动
执行 sigmoid 或任务名映射。

## 一致性

Rust 和 Python 各自解析并校验同一 schema，共享接受/拒绝 fixtures。规范化过程展开
默认值、稳定排序节点，并以统一浮点表示生成 canonical JSON，供后续 manifest 保存
完整契约及摘要。标签和 mask 最终必须与特征配置联合校验；原生契约引用的 label 不
允许引用 feature/discard source。当前 `Trainer` 已校验 label 必须是 `role: label`，
mask source 必须存在；“label 不配置默认值”仍受现有 `FlowConfig` 兼容格式限制，尚未
在实际训练配置中强制。

## 兼容与迁移

原生 `output_contract` 与旧 `tasks/task_config/label_col_map/metrics` 禁止混用。
迁移按以下阶段推进：

1. 落地 schema、双端校验和共享 fixtures。
2. 实现通用 `OutputHead`、`ModelExecution` 和 `ObjectiveEngine`。
3. 先迁移标准 ESMM，再迁移其他 shared-backbone 模型。
4. 增加 MMoE 多命名表示支持。
5. 将规范化契约写入 manifest，并切换 serving 输出投影。
6. 提供旧配置机械适配器，完成配置迁移后删除旧执行路径。

当前进度：

- 阶段 1、2 已完成。
- 标准 `esmm` 已完成阶段 3，示例见 `examples/models/esmm_output_contract.yaml`。
- `gdcn_esmm`、UniMixer、TokenMixer-Large、RankMixer 和 MMoE 仍使用
  `tasks + task_config` 兼容路径。
- contract ESMM 的 `forward()` 已只返回公开输出，`forward_execution()` 同时保留内部
  节点供训练和评估使用；manifest 仍未保存规范化契约。
