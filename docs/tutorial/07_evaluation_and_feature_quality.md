# 07. 训练评估与特征质量

[目录](README.md) | [上一章](06_model_structure_and_weight_binding.md) | [下一章](08_artifact_publish_and_versioning.md)

这个系统里，训练评估不是只看一个总 loss。

你需要同时关心：

- 模型是否真的学到了东西。
- 各 task 的指标是否正常。
- 特征预处理有没有大量缺失、默认值、空序列或异常 hash 分布。

## 评估看什么

`python/src/train/training/eval/evaluator.py` 会按 `task × metric` 计算评估结果。

legacy 模型的默认指标来自 `TrainConfig.eval.metrics`，也可以在 `tasks[].metrics` 里逐
task 覆盖。原生模型只计算 `output_contract.metrics` 显式声明的指标。

常见指标包括：

- `auc`
- `logloss`
- `mae`
- `mse`
- `gauc`

`monitor_task` 和 `monitor_metric` 决定 early stopping 和 best checkpoint 看哪一个值。
原生契约中，`monitor_task` 使用 metric 的 `source` 节点名，例如
`ctcvr_prob`，不是公开输出名 `ctcvr`。

## 训练 loss 是怎么合成的

`MultiTaskLoss` 支持三种加权方式：

- `equal`
- `static`
- `uncertainty`

其中 `static` 是最常用的默认值：每个 task 按配置里的 weight 累加。

对 `stay_time_label` 这种特殊标签，会走 `weighted_bce_stay`，不是普通 BCE。

原生契约使用 `ObjectiveEngine`，当前要求 `loss_weighting: static`。每个 objective
通过自身的 `weight` 加权，支持：

- `binary_cross_entropy_with_logits`：只接受 `binary_logit`
- `binary_cross_entropy`：只接受 `probability`，计算前按 epsilon 截断
- `mse`、`mae`、`huber`：接受 `regression` 或 `score`
- `weighted_bce_stay`：接受 `binary_logit`

`auc` 可读取 logit 或 probability；logit 会转换为概率。`logloss` 要求 probability，
已经是 probability 的节点不会再次 sigmoid。

## feature quality 在看什么

训练时 `Trainer` 会从验证 batch 里抽样，生成 `FeatureQualityReport`。

它会记录三类统计：

### source 质量

- `missing_rate`
- `default_rate`

### embeddable feature 质量

- `empty_sequence_rate`
- `truncation_rate`
- `mean_length`
- `padding_rate`
- `bucket_utilization`

### hash cache 质量

如果某些 hash 算子启用了缓存，还会记录：

- `total`
- `hit_rate`
- `miss_rate`
- `cache_size`

这些指标最后会进入 run manifest，方便你回头判断是模型问题还是数据问题。

## 怎样解读常见异常

几个经验判断：

- `missing_rate` 高，通常是数据接入或列对齐问题。
- `default_rate` 高，但 `missing_rate` 不高，通常是原始值大量落到默认值。
- `empty_sequence_rate` 高，通常是序列解析或上游字段缺失。
- `padding_rate` 高，通常是序列长度设太长或实际行为太稀疏。
- `bucket_utilization` 过低，通常是 hash 空间过大或样本量太小。

## 训练日志里重点看什么

建议至少看这几类输出：

- `data: files=... rows(total=... train=... eval=...)`
- `optimizer: ... warmup=...`
- `epoch ... loss=...`
- `task_metric` 明细
- `feature quality: rows=...`

如果训练过程里 loss 看起来正常，但 feature quality 极差，那问题通常不在模型，而在数据契约。

## 评估失败的典型原因

1. 任务定义和 label 列不一致。
2. 验证集没有有效标签。
3. 某个 task 在 model 里没输出。
4. `gauc_group_feature` 不存在或分组字段异常。
5. 训练文件里标签列被当成 feature 读进去了。

## 建议的排查顺序

1. 先看 `tasks`，或原生契约的 `objectives/metrics` 和 label 列。
2. 再看 `feature quality`。
3. 再看每个 task 的 metric。
4. 最后再调 loss weighting、early stopping 或学习率。

下一章讲产物发布和版本管理。那里会把 checkpoint、best/latest 别名、运行记录和 serving manifest 串成一套可回滚的发布目录。
