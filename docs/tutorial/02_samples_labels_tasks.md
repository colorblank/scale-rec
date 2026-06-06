# 02. 样本表、标签与任务定义

推荐排序模型训练的第一步不是选模型，而是定义一行样本代表什么、哪些字段是特征、哪些字段是监督标签、每个标签对应哪个任务。scale-rec 的 discover 示例采用典型的 pointwise 排序样本：

```text
一行样本 = 一个 user 在一个 context 下对一个 item 的一次曝光/候选打分记录
```

训练时每行样本独立进入模型；在线推理时可以用 `/predict` 逐行打分，也可以用 `/predict/broadcast` 固定 user/context，对多个 item 批量打分。

## 一行样本的结构

`examples/feature_config_discover.yaml` 的 `sources` 定义了样本列。每个 source 可以带 `source` 字段，用来标记它属于 Item、User 还是 Context：

```yaml
- name: item_id
  source: Item
  dtype: int
  default_val: '0'

- name: user_id
  source: User
  dtype: int
  default_val: '0'

- name: scene
  source: Context
  dtype: int
  default_val: '0'
```

这三个分组的含义：

| 分组 | 含义 | 示例 |
|---|---|---|
| Item | 被排序的内容、商品或资产侧字段 | `item_id`、`item_type`、`title`、`quality_score_label`、`stock_list` |
| User | 用户画像、行为序列、偏好字段 | `user_id`、`fav_securities`、`interest_keywords`、`follow_authors` |
| Context | 本次请求或曝光上下文 | `scene`、`rec_algo`、`p_date`、`stay_time` |
| Label | 训练监督信号，不进入在线推理特征 | `is_click`、`is_cvr`、`is_click_detail`、`is_click_stock`、`stay_time_label` |

从建模角度看，Item/User/Context 都是特征输入；Label 是训练目标。在线请求必须提供足够的 Item/User/Context 字段，不能依赖 Label。

## 字段角色

feature config 里通过 `role` 区分字段用途：

```yaml
- name: is_click
  dtype: int
  default_val: '0'
  role: label

- name: stay_time_label
  dtype: int
  default_val: '-1'
  role: label
```

默认 `role` 是 `feature`。常见角色：

| role | 用途 |
|---|---|
| `feature` | 原始特征列，可被 operators 消费并进入模型 |
| `label` | 监督标签列，被训练 loss/metric 使用，不作为模型输入 |
| `discard` | 读取后丢弃，用于兼容数据格式或保留非建模字段 |

一个字段是不是训练标签，首先由 `role: label` 标出来；这个标签是否真的参与 loss，还要看 model config 里的 `tasks`。

## 当前 discover 标签

当前 discover 示例定义了这些 label source：

| 标签列 | 类型 | 默认值 | 语义 |
|---|---|---|---|
| `is_click` | int | `0` | 是否点击 |
| `is_cvr` | int | `0` | 是否转化 |
| `is_click_detail` | int | `0` | 点击后是否进入详情 |
| `is_click_stock` | int | `0` | 点击后是否关注/查看股票相关行为 |
| `stay_time_label` | int | `-1` | 停留时长标签，训练中用 `weighted_bce_stay` 处理 |
| `ctr` | int | `0` | 兼容/派生标签，当前 GDCN+ESMM tasks 未直接使用 |
| `cvr` | int | `0` | 兼容/派生标签，当前 GDCN+ESMM tasks 未直接使用 |

注意：`ctr` 和 `cvr` 虽然在 feature config 里是 label，但当前 `examples/model_gdcn_esmm.yaml` 的 `tasks` 没有引用它们，因此不会进入当前模型的 loss。是否参与训练，以 model config 的 `tasks` 为准。

## 任务定义

`examples/model_gdcn_esmm.yaml` 把模型输出、标签列、loss 和 metric 绑定在一起：

```yaml
tasks:
  - {name: click, label: is_click, loss: bce, metrics: [auc, logloss]}
  - {name: cvr, label: is_cvr, loss: bce, metrics: [auc, logloss]}
  - {name: detail, label: is_click_detail, loss: bce, metrics: [auc, logloss]}
  - {name: stock, label: is_click_stock, loss: bce, metrics: [auc, logloss]}
  - {name: stay, label: stay_time_label, loss: weighted_bce_stay, metrics: [mae, mse]}
```

这里有四层含义：

| 字段 | 含义 |
|---|---|
| `name` | 模型输出任务名，也是 loss 查找输出的 key |
| `label` | batch 里的标签列名 |
| `loss` | 该任务使用的损失函数 |
| `metrics` | 评估时记录的指标 |

训练时 `MultiTaskLoss` 会遍历模型输出，跳过 `ct*` 这类关系输出，再按 task spec 找对应 label 列并计算 loss。

## ESMM 关系任务

GDCN+ESMM / ESMM 不只输出基础任务，还会定义关系输出：

```yaml
task_config:
  relations:
    - {target: ctcvr, sources: [click, cvr], op: multiply}
    - {target: ctdetail, sources: [click, detail], op: multiply}
    - {target: ctstock, sources: [click, stock], op: multiply}
    - {target: ctstay, sources: [detail, stay], op: multiply}
```

这些关系输出用于表达条件概率链路：

```text
P(ctcvr)    = P(click)  * P(cvr | click)
P(ctdetail) = P(click)  * P(detail | click)
P(ctstock)  = P(click)  * P(stock | click)
P(ctstay)   = P(detail) * P(stay | detail)
```

训练 loss 默认只对基础任务 `click/cvr/detail/stock/stay` 计算；`ctcvr/ctdetail/ctstock/ctstay` 是模型前向输出中的派生概率，用于评估或业务排序时选择合适的分数。

## demo 标签生成与真实业务标签

`examples/discover_label_policy.yaml` 只服务于 demo 数据生成。它定义合成数据里点击、详情、股票、转化、停留时长的规则，例如：

```yaml
click:
  quality_weight: 0.45
  threshold: 0.42

cvr:
  quality_threshold: 0.68
  stay_time_threshold: 240
```

生成脚本 `python/src/scale_rec_demo/generate_discover_data.py` 会根据这些规则写出 `python/artifacts/demo/discover_train_data.txt`。

真实业务接入时，不应该复用 demo label policy 来定义线上标签。生产标签通常来自曝光日志、点击日志、转化日志、停留时长日志和归因规则。你需要保证最终训练文件里有 model config `tasks` 引用的 label 列即可。

## 训练文件与 header

discover demo 默认输出 TSV，无 header。训练命令因此需要加：

```bash
--no-header
```

无 header 时，读取逻辑会按 `feature_config_discover.yaml` 的 source 顺序解释每一列。这个顺序就是训练文件的 schema。如果生产数据列顺序不同，要么调整导出顺序，要么提供带 header 的文件并去掉 `--no-header`。

常见错误：

- label 列缺失：训练报 `No supervised batches were processed` 或评估 labels 为空。
- 列顺序错位：某些 source 的 `missing_rate/default_rate` 异常升高。
- label 名称不一致：model config 的 `tasks[].label` 找不到对应列。
- 把 label 当 feature 使用：在线推理时没有该字段，导致训练和推理分布不一致。

## batch 中 labels 如何进入 loss

训练 batch 大致分成两部分：

```text
batch["features"] -> dag.preprocess_batch(...) -> model inputs
batch["labels"]   -> MultiTaskLoss 按 tasks[].label 读取
```

模型只看到 `features` 预处理后的 LongTensor；loss 函数看到 `labels`。这条边界很重要：label 列可以存在于原始 TSV 中，但不能被 feature operators 作为输入消费。

`stay_time_label` 是一个特殊例子：它是连续时长标签，但当前任务配置使用 `weighted_bce_stay`，训练代码会用专门逻辑处理它。因此新增类似任务时，需要同时确认 loss 函数是否支持该标签语义。

## 接入真实数据的检查清单

接入生产样本前，至少确认这些事项：

1. 一行样本是否代表一个 user-item-context 打分样本。
2. Item/User/Context 字段是否能满足在线推理请求构造。
3. label 列是否全部在 feature config 中标为 `role: label`。
4. model config 的 `tasks[].label` 是否都能在训练文件中找到。
5. 无 header 文件的列顺序是否严格等于 feature config 的 source 顺序。
6. 训练用 label 没有被 operators 当作 feature 输入。
7. 多日训练时，最后日期文件的 label 分布适合作为验证集。
8. demo label policy 没有被误当成生产标签定义。

下一章会进入特征工程契约，说明这些原始字段如何通过 DAG、hash、序列处理和 embedding 配置变成模型输入。
