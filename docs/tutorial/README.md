# scale-rec 推荐排序系统教程

本教程面向要把 scale-rec 用作推荐排序模型训练和在线推理系统的工程师。内容按真实排序系统的建设顺序组织：先明确样本、标签和特征契约，再训练多任务排序模型，最后导出权重并部署 Rust 在线推理服务。

教程不按源码目录展开，而按“离线训练 -> 产物发布 -> 在线推理 -> 质量与性能治理”的链路展开。源码目录只在需要定位实现时作为索引出现。

## 教程主线

```text
原始日志/样本
    │
    ▼
样本表与标签定义
    │
    ▼
特征配置与 DAG 预处理
    │
    ▼
多任务排序模型训练
    │
    ▼
评估、feature quality、debug
    │
    ▼
safetensors + serving manifest
    │
    ▼
Rust HTTP 排序推理服务
    │
    ▼
压测、版本管理、线上排障
```

## 章节规划

| 章节 | 目标 | 关键文件 | 状态 |
|---|---|---|---|
| [01. 排序系统全链路架构](01_project_structure.md) | 建立训练、配置、导出、在线推理的整体心智模型 | `README.md`、`python/src/train`、`src/server`、`examples` | 已添加 |
| [02. 样本表、标签与任务定义](02_samples_labels_tasks.md) | 说明一行样本如何对应 user-item-context，click/cvr/detail/stock/stay 等任务如何进入 loss | `examples/shared/feature_config_discover.yaml`、`examples/shared/discover_label_policy.yaml`、`examples/models/gdcn_esmm.yaml` | 已添加 |
| [03. 特征工程契约](03_feature_contract.md) | 讲 sources、operators、embedding、role、DAG、hash 空间、序列 padding 和 Python/Rust 一致性 | `examples/shared/feature_config_discover.yaml`、`python/src/train/core/dag.py`、`src/feats/dag.rs` | 已添加 |
| 04. 离线训练流程 | 跑通 demo 和生产流式训练，解释 batch、eval、checkpoint、early stopping、EMA | `python/src/train/app/main.py`、`python/src/train/training/trainer.py`、`examples/shared/train_defaults.yaml` | 待添加 |
| 05. 多日训练与增量微调 | 讲 `--data-glob`、日期闭区间、最后日期验证集、`--init-weights` 微调语义 | `python/src/train/app/cli.py`、`python/src/train/app/data.py` | 待添加 |
| 06. 模型结构与权重绑定 | 讲 LR baseline、DeepFM/MMoE/ESMM/GDCN+ESMM/UniMixer/TokenMixer-Large/RankMixer、任务塔、safetensors key 与 Candle 路径 | `python/src/train/models`、`src/models`、`python/src/train/app/export.py` | 待添加 |
| 07. 训练评估与特征质量 | 讲 loss、AUC、回归指标、feature quality、序列 padding 空值率、bucket 利用率 | `python/src/train/training/metrics`、`python/src/train/training/quality.py` | 待添加 |
| 08. 产物发布与版本管理 | 讲 run manifest、serving manifest、sha256、model version、publish path、回滚策略 | `python/src/train/app/manifest.py`、`src/server/manifest.rs` | 待添加 |
| 09. Rust 在线推理服务 | 讲 model registry、`/predict`、`/predict/broadcast`、多版本加载、fallback version | `src/server`、`docs/API.md` | 待添加 |
| 10. 性能优化与大文件训练 | 讲 50GB/日数据、pandas chunk、memory map、fast-no-na、流式 eval、服务压测 | `python/src/train/app/data.py`、`src/bin/bench.rs` | 待添加 |
| 11. Debug 与一致性验证 | 讲单样本 trace、batch tensor、Python/Rust golden consistency、权重 key 排查 | `python/src/train/debug`、`tests/golden_consistency.rs` | 待添加 |

## 推荐阅读路径

如果目标是“先跑起来”，按 01 -> 02 -> 04 -> 08 -> 09。

如果目标是“改特征并保证线上一致”，按 01 -> 03 -> 07 -> 11。

如果目标是“接入生产多日训练”，按 01 -> 02 -> 05 -> 07 -> 10。

如果目标是“新增模型或改模型结构”，按 01 -> 06 -> 08 -> 11。

## 与参考文档的关系

教程负责给出推荐排序系统的工程路径；参考文档负责查细节：

- [训练手册](../TRAINING_GUIDE.md)：训练命令、数据参数、保存发布和服务加载。
- [特征算子](../feature_operators.md)：16 个特征算子的参数、输入输出和边界行为。
- [HTTP API](../API.md)：HTTP 请求、响应、错误码和批量推理格式。
- [开发环境](../DEVELOPMENT.md)：本地测试、格式化和端到端验证命令。
- [Docker 打包](../../docker/README.md)：容器构建和模型挂载方式。
