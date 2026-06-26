# Tutorials

教程面向学习路径，按推荐系统从离线样本到在线推理的建设顺序组织。正文统一维护在 `docs/tutorials/`；旧 `docs/tutorial/` 目录只保留兼容跳转页。

## Recommended paths

如果目标是先跑起来：

1. [排序系统全链路架构](end_to_end_recommendation.md)
2. [样本表、标签与任务定义](samples_labels_and_tasks.md)
3. [离线训练流程](offline_training.md)
4. [产物发布与版本管理](artifact_publishing.md)
5. [Rust 在线推理服务](rust_inference_service.md)

如果目标是改特征并保证线上一致：

1. [排序系统全链路架构](end_to_end_recommendation.md)
2. [特征工程契约](feature_dag.md)
3. [训练评估与特征质量](evaluation_and_feature_quality.md)
4. [Debug 与一致性验证](debugging_consistency.md)

如果目标是新增模型或修改模型结构：

1. [排序系统全链路架构](end_to_end_recommendation.md)
2. [模型结构与权重绑定](model_and_weight_binding.md)
3. [产物发布与版本管理](artifact_publishing.md)
4. [Debug 与一致性验证](debugging_consistency.md)

## Tutorials

| Tutorial | Goal |
|---|---|
| [End-to-end Recommendation Pipeline](end_to_end_recommendation.md) | 建立训练、配置、导出、在线推理的整体心智模型 |
| [Samples, Labels, and Tasks](samples_labels_and_tasks.md) | 理解样本行、标签列和 output_contract |
| [Feature DAG](feature_dag.md) | 理解 sources、operators、embedding、role、DAG 和 Python/Rust 一致性 |
| [Offline Training](offline_training.md) | 跑通 demo 和生产流式训练 |
| [Multi-day Training](multi_day_training.md) | 使用 `--data-glob`、日期范围、`--eval-data` 和 `--init-weights` |
| [Model Structure and Weight Binding](model_and_weight_binding.md) | 理解模型 registry、OutputHead、safetensors key 和 Candle 路径 |
| [Evaluation and Feature Quality](evaluation_and_feature_quality.md) | 理解 loss、metrics、feature quality 和 bucket hit count |
| [Artifact Publishing](artifact_publishing.md) | 理解 run manifest、serving manifest、sha256、model version 和回滚 |
| [Rust Inference Service](rust_inference_service.md) | 理解 model registry、`/predict`、`/predict/broadcast` 和 fallback version |
| [Performance Tuning](performance_tuning.md) | 理解 pandas chunk、memory map、fast-no-na、prefetch 和压测 |
| [Debugging and Consistency](debugging_consistency.md) | 排查单样本 trace、batch tensor、Python/Rust mismatch 和权重 key |

## Legacy path mapping

旧 `docs/tutorial/` 路径保留用于兼容外部链接；正文已合并到下列新教程页：

| Legacy path | Canonical tutorial |
|---|---|
| [01. 排序系统全链路架构](../tutorial/01_project_structure.md) | [End-to-end Recommendation Pipeline](end_to_end_recommendation.md) |
| [02. 样本表、标签与任务定义](../tutorial/02_samples_labels_tasks.md) | [Samples, Labels, and Tasks](samples_labels_and_tasks.md) |
| [03. 特征工程契约](../tutorial/03_feature_contract.md) | [Feature DAG](feature_dag.md) |
| [04. 离线训练流程](../tutorial/04_offline_training_flow.md) | [Offline Training](offline_training.md) |
| [05. 多日训练与增量微调](../tutorial/05_multi_day_incremental.md) | [Multi-day Training](multi_day_training.md) |
| [06. 模型结构与权重绑定](../tutorial/06_model_structure_and_weight_binding.md) | [Model Structure and Weight Binding](model_and_weight_binding.md) |
| [07. 训练评估与特征质量](../tutorial/07_evaluation_and_feature_quality.md) | [Evaluation and Feature Quality](evaluation_and_feature_quality.md) |
| [08. 产物发布与版本管理](../tutorial/08_artifact_publish_and_versioning.md) | [Artifact Publishing](artifact_publishing.md) |
| [09. Rust 在线推理服务](../tutorial/09_rust_inference_service.md) | [Rust Inference Service](rust_inference_service.md) |
| [10. 性能优化与大文件训练](../tutorial/10_performance_and_large_files.md) | [Performance Tuning](performance_tuning.md) |
| [11. Debug 与一致性验证](../tutorial/11_debug_and_consistency.md) | [Debugging and Consistency](debugging_consistency.md) |
