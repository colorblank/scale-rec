# 01. 排序系统全链路架构

scale-rec 的定位不是单独的训练脚本，也不是单独的推理服务，而是一套推荐排序模型从离线训练到在线推理的端到端系统。Python 侧负责训练和导出，Rust 侧负责低延迟推理，两端通过同一份特征配置、模型配置和 safetensors 权重绑定在一起。

## 一句话架构

```text
Python: 样本读取 + 特征 DAG + PyTorch 训练 + safetensors/manifest 导出
Rust:   同构特征 DAG + Candle 模型加载 + HTTP 排序推理服务
```

这套架构的核心约束是：离线训练看到的特征、在线推理计算的特征、模型结构和权重命名必须一致。任何一处不一致，都会变成线上离线不一致、权重加载失败或排序分数异常。

## 推荐排序链路

```text
1. 样本与标签
   user / item / context / label 进入 discover TSV

2. 特征契约
   feature_config_discover.yaml 定义 sources、operators、embed、role

3. 离线训练
   Python FeatureDag 预处理样本，PyTorch 模型训练多任务排序目标

4. 训练评估
   计算 loss、AUC、回归指标、feature quality、序列 padding 和 hash bucket 使用情况

5. 产物导出
   导出 safetensors 权重、run manifest、serving manifest

6. 在线加载
   Rust server 扫描 manifest，加载 feature config、model config 和权重

7. 排序推理
   HTTP /predict 或 /predict/broadcast 输入 user/item/context，返回多任务分数
```

后续章节会逐步展开每一步。本章只建立整体结构和代码定位方式。

## 配置是系统契约

scale-rec 里最重要的不是某个模型类，而是 `examples/` 下的配置契约：

```text
examples/
├── models/                          # 按模型拆分的配置目录
│   ├── lr.yaml                      # LR 单目标基线配置
│   ├── gdcn_esmm.yaml               # GDCN+ESMM 模型配置
│   ├── unimixer.yaml                # UniMixer 模型配置
│   ├── token_mixer_large.yaml       # TokenMixer-Large 模型配置
│   └── rankmixer.yaml               # RankMixer 模型配置
└── shared/                          # 共享配置目录
    ├── feature_config_discover.yaml # 特征契约：原始列、DAG、embedding、标签 role
    ├── train_defaults.yaml          # 训练策略：batch、optimizer、eval、early stopping
    └── discover_label_policy.yaml   # demo 样本标签生成规则
└── gen_discover_config.py           # 生成 discover 特征配置的脚本
```

配置职责需要分清：

- `shared/feature_config_discover.yaml` 是训练和推理共享的特征协议。它决定原始字段如何变成模型输入 tensor。
- `models/*.yaml` 是模型语义协议。它决定模型类型、任务列表、label 映射、loss 和 metric。`lr.yaml` 提供了一个最小的单目标二分类基线，其它模型则在此基础上扩展多任务或更复杂的结构。
- `shared/train_defaults.yaml` 是训练运行策略。它不应该影响在线推理结果。
- `shared/discover_label_policy.yaml` 只用于 demo 数据生成，不是线上协议。

当你要改一个排序特征时，优先思考它属于哪个层次：原始列、特征算子、embedding 空间、模型输入维度，还是任务定义。不要把这些职责混在一个文件里。

## Python 训练侧分层

```text
python/src/train/
├── app/          # 训练 CLI、数据读取、artifact、manifest、权重导出
├── core/         # FlowConfig、FeatureDag、schema、task 定义
├── ops/          # Python 特征算子
├── layers/       # PyTorch 基础层
├── models/       # 推荐排序模型
├── training/     # Trainer、loss、metrics、eval、feature quality
└── debug/        # 特征预处理 trace
```

训练请求进入后的关键调用链：

```text
train.app.main
  -> train.app.cli            # 解析 CLI、合并 train/model 配置、解析多日数据路径
  -> train.app.data           # pandas 读取 TSV，流式生成 batch
  -> train.core.dag           # 执行特征 DAG，生成 LongTensor 输入
  -> train.models             # 构建 PyTorch 排序模型
  -> train.training.trainer   # 训练、评估、checkpoint
  -> train.app.export         # 导出 safetensors
  -> train.app.manifest       # 写 run/serving manifest
```

排查训练问题时，先确认是哪一层出错：

- 数据读不到：看 `app/data.py` 和 CLI 参数。
- 特征不对：看 `core/dag.py`、`ops/` 和 feature config。
- loss/metric 不对：看 model config 的 tasks 和 `training/metrics`。
- 权重不能被 Rust 加载：看 `app/export.py` 和 Rust `VarBuilder::pp()` 路径。

## Rust 推理侧分层

```text
src/
├── feats/         # Rust 特征配置、DAG、算子
├── layers/        # Candle 网络层
├── models/        # Rust 推荐排序模型
├── server/        # 模型注册、manifest 加载、HTTP routes、推理 engine
└── bin/           # server、bench、demo_inference
```

在线服务启动后的关键调用链：

```text
src/bin/server.rs
  -> server::registry         # 扫描 model-dir 或加载 model-path
  -> server::manifest         # 解析 serving manifest
  -> feats::config/dag        # 加载同一份 feature config
  -> models::build            # 按 model config 构建 Candle 模型
  -> server::engine           # 执行特征预处理和模型 forward
  -> server::routes           # 暴露 /predict、/predict/broadcast
```

Rust 侧不重新训练，也不重新定义特征。它只消费 Python 发布出来的配置和权重。

## 在线离线一致性的边界

这个项目通过三类文件维持一致性：

| 文件 | 离线训练 | 在线推理 | 不一致时的典型问题 |
|---|---|---|---|
| feature config | Python `FeatureDag` 读取 | Rust `FeatureDag` 读取 | 同一请求在线离线特征值不同 |
| model config | PyTorch 模型构建 | Candle 模型构建 | 结构不一致，权重 shape/key 加载失败 |
| safetensors | Python 导出 | Rust 加载 | 权重 key 不匹配或分数异常 |

因此，任何涉及双端共享逻辑的修改都必须验证：

```bash
cargo test --test model_smoke

PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

如果只是修改 Python 训练策略，例如 batch size、early stopping、学习率、checkpoint 保留数量，一般不需要改 Rust。

## 训练产物流

一次训练会产生两类产物：

```text
python/artifacts/demo/
└── model_gdcn_esmm/<run_version>/
    ├── checkpoints/                     # 训练 checkpoint
    ├── serving/
    │   ├── model.safetensors            # 发布权重
    │   ├── model.manifest.yaml          # serving manifest，给 Rust 服务加载
    │   └── configs/                     # 本次训练使用的 feature/model config 副本
    └── run.manifest.yaml                # 训练 run 元数据
```

生产加载推荐使用 serving manifest，而不是裸 safetensors。manifest 会绑定：

- 模型名和版本
- 权重文件
- feature config
- model config
- sha256
- task、label、metric 等训练元数据

这能避免线上服务加载了错误配置或错误权重。

## 排序推理形态

服务提供两种核心推理形态：

- `/predict`：pointwise 预测，每行样本包含完整 user/item/context。
- `/predict/broadcast`：一个 user/context 对多个 item 批量打分，适合召回后排序。

推荐排序系统里更常见的是 broadcast：一次请求里固定用户侧特征，对候选 item 列表打分，再按 click/cvr/detail/stock/stay 等任务输出排序分数。

接口细节见 [HTTP API](../API.md)。

## 代码目录只作为定位工具

顶层目录可以这样理解：

```text
scale-rec/
├── python/       # 训练、导出、Python 测试
├── src/          # Rust 在线推理
├── examples/     # 离线和在线共享配置
├── docs/         # 教程和参考文档
├── tests/        # Rust 集成测试
└── docker/       # 部署打包
```

更详细的开发命令见 [开发环境](../DEVELOPMENT.md)。

## 后续章节怎么读

如果你要接入真实业务数据，下一步先读“样本表、标签与任务定义”，弄清一行样本如何表达 user-item-context 和多任务标签。

如果你要改特征，跳到“特征工程契约”，重点看 hash 空间、序列特征、padding、embedding pooling 和 Python/Rust 一致性。

如果你要上线服务，按“离线训练流程 -> 产物发布与版本管理 -> Rust 在线推理服务”的顺序读。
