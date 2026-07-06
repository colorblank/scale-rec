# Model Zoo

本文档汇总当前仓库已注册模型的定位、输入形式、关键参数和示例配置。所有模型均通过
`python/src/train/models/__init__.py` 与 `src/models/mod.rs` 注册，训练侧和 Rust 推理侧
必须保持同名结构与权重 key 对齐。

## 模型总览

| Model type | 论文/arXiv | 示例配置 | 模型类别 | 输入表示 | 核心结构 | 输出表示 | 适用场景 |
|---|---|---|---|---|---|---|---|
| `lr` | 工程 baseline，无对应 arXiv | `examples/models/lr.yaml` | 线性 baseline | feature embedding concat | 一阶线性打分 | `shared` 标量 logit | 低成本 baseline、特征连通性验证、线上回退 |
| `deepfm` | [DeepFM, arXiv:1703.04247](https://arxiv.org/abs/1703.04247) | `examples/models/deepfm.yaml` | FM + DNN | feature embedding concat | FM 一阶/二阶交互 + MLP | `shared` 标量 logit | 稀疏 ID 特征为主、需要显式二阶交互的 CTR 任务 |
| `mmoe` | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007) | `examples/models/mmoe.yaml` | 多任务专家模型 | feature embedding concat | shared bottom + experts + task gates | 每个 tower 可指定命名 representation，如 `click_rep` | 多任务目标差异明显、希望 gate 学习任务差异 |
| `esmm` | [ESMM, arXiv:1804.07931](https://arxiv.org/abs/1804.07931) | `examples/models/esmm_output_contract.yaml` | 多任务转化率模型 | feature embedding concat | shared bottom + 多任务 tower + 概率关系 | `shared` | CTR/CVR/CTCVR 等有级联关系的任务 |
| `gdcn_esmm` | [GDCN, arXiv:2311.04635](https://arxiv.org/abs/2311.04635) + [ESMM, arXiv:1804.07931](https://arxiv.org/abs/1804.07931) | `examples/models/gdcn_esmm.yaml` | GDCN + ESMM | feature embedding concat | gated cross network + deep branch + ESMM tower | `shared` | 需要显式 cross 特征交互，同时保留 ESMM 概率关系 |
| `pepnet` | [PEPNet, arXiv:2302.01115](https://arxiv.org/abs/2302.01115) | `examples/models/pepnet.yaml` | 个性化门控模型 | feature embedding concat + prior feature 分组 | EPNet/PPNet gate + task towers | `shared` | 需要按用户、物品、场景 prior 做个性化调制 |
| `unimixer` | [UniMixer, arXiv:2604.00590](https://arxiv.org/abs/2604.00590) | `examples/models/unimixer.yaml` | token mixing 排序模型 | 外部 `FeatureTokenizer` token 序列 | UniMixer / UniMixerLite token interaction | `shared` | 中等规模 token 化 sparse 特征，需要稳定 token 交互 |
| `token_mixer_large` | [TokenMixer-Large, arXiv:2602.06563](https://arxiv.org/abs/2602.06563) | `examples/models/token_mixer_large.yaml` | 大规模 token mixer | 外部 `FeatureTokenizer` token 序列 | Mixing & Reverting + per-token SwiGLU | `shared` | token 数和 token_dim 较大、需要更强 token mixing 的排序模型 |
| `rankmixer` | [RankMixer, arXiv:2507.15551](https://arxiv.org/abs/2507.15551) | `examples/models/rankmixer.yaml` | dense token mixer | 外部 `FeatureTokenizer` token 序列 | RankMixer block + per-token FFN | `shared` | 排序 token 结构明确、希望保留轻量 dense mixer |
| `rankup` | [RankUp, arXiv:2604.17878](https://arxiv.org/abs/2604.17878) | `examples/models/rankup.yaml` | RankUp sparse feature scaling | 内置 RankUp tokenizer | 随机稀疏特征分组 + multi-embedding + global/cross/task token + RankMixer block | `shared`，以及 `task_0`、`task_1` 等 task token 表示 | Top-K sparse 特征较多，需要扩大稀疏特征容量并使用 task-specific 表示 |
| `hyformer` | [HyFormer, arXiv:2601.12681](https://arxiv.org/abs/2601.12681) | `examples/models/hyformer.yaml` | hybrid sequence/query model | 内置 HyFormer tokenizer | global query generation + sequence memory cross-attention + RankMixer-style query boosting | `shared` | 同时有非序列特征和序列行为特征，需要用查询 token 解码序列记忆 |
| `uniformer` | [UniFormer, arXiv:2606.27058](https://arxiv.org/abs/2606.27058) | `examples/models/uniformer.yaml` | feature/task interaction model | 内置 UniFormer tokenizer | FIM feature interaction + TIM task-token interaction | `shared`，以及 `task_0`、`task_1` 等 task token 表示 | 同时需要建模 sequence/non-sequence feature 交互和 task-specific 表示 |

## 选型建议

| 目标 | 推荐模型 | 原因 |
|---|---|---|
| 先验证训练链路、特征配置、导出和 Rust serving | `lr`、`deepfm` | 参数少，失败时更容易定位是特征、标签还是模型问题 |
| 单目标 CTR，稀疏 ID 特征为主 | `deepfm`、`gdcn_esmm` | DeepFM 提供二阶 FM 交互；GDCN 显式建模 gated cross |
| CTR/CVR/详情/收藏/停留等多任务排序 | `esmm`、`gdcn_esmm` | 原生支持 tower + relation 的概率图，例如 `ctcvr_prob = click_prob * cvr_prob` |
| 多任务目标差异大，任务之间共享不完全一致 | `mmoe` | expert + gate 可以按任务选择不同共享表示 |
| 需要场景、用户、物品 prior 做个性化门控 | `pepnet` | `ep_prior_features`、`pp_prior_features` 明确控制 gate 输入 |
| 希望把 sparse feature 组织成 token 序列 | `unimixer`、`token_mixer_large`、`rankmixer` | 三者共用外部 `FeatureTokenizer`，适合 token 化特征建模 |
| sparse 特征数量增长，需要扩大 Top-K 特征容量 | `rankup` | 使用随机稀疏分组、多 embedding table、global/cross/task token |
| 行为序列特征是核心信号 | `hyformer` | 使用 query token 对 sequence memory 做 cross-attention，再做 query boosting |
| 多任务需要显式 task token 表示 | `uniformer` | TIM 生成 `task_i` 表示，可在 output_contract tower 中直接绑定 |

## 配置与输出契约

所有 `examples/models/*.yaml` 示例均使用 `output_contract.version: 1`。推荐新增模型配置也使用
原生 output contract，不再新增 legacy `tasks/label_col_map/metrics` 组合。

| 配置块 | 作用 | 说明 |
|---|---|---|
| `type` | 模型 registry key | 必须是上表中的 model type |
| 模型私有参数 | 控制 backbone 结构 | 例如 `token_dim`、`num_tokens`、`num_blocks`、`d`、`d_ff` |
| `output_contract.graph.towers` | 定义 task tower | tower 的 `input` 默认是 `shared`；RankUp 和 UniFormer 可使用 `task_0` 等表示 |
| `output_contract.graph.relations` | 定义输出关系图 | 支持 `sigmoid`、`multiply`、`add`、`identity` |
| `output_contract.objectives` | 定义训练 loss | 可引用内部 tower 或 relation 节点 |
| `output_contract.metrics` | 定义评估指标 | 可与 objectives 使用不同节点 |
| `output_contract.outputs` | 定义 serving 公开输出 | Rust serving 只暴露这里声明的稳定输出 |

## 关键参数

| Model type | 关键参数 | 约束与注意事项 |
|---|---|---|
| `deepfm` | `fm_k`、`deep_hidden_dims` | `fm_k` 影响二阶 FM embedding 维度，改变后旧权重不可直接复用 |
| `mmoe` | `shared_bottom_dims`、`num_experts`、`expert_hidden_dims`、`expert_output_dim` | `graph.towers[].input` 必须能映射到 MMoE 生成的 representation |
| `esmm` | `shared_bottom_dims` | 级联概率关系建议写在 `output_contract.graph.relations` 中 |
| `gdcn_esmm` | `cross_layers`、`deep_hidden_dims`、`shared_bottom_dims` | `cross_layers` 增加会改变权重结构 |
| `pepnet` | `prior_dim`、`ep_prior_features`、`pp_prior_features`、`deep_hidden_dims`、`shared_bottom_dims` | prior 特征必须来自可 embedding 的特征名 |
| `unimixer` | `token_dim`、`num_tokens`、`num_blocks`、`block_size`、`use_lite`、`hidden_factor`、`num_basis`、`rank`、`use_siamese` | 需要外部 `FeatureTokenizer`；`total_embed_dim` 需要能按 `num_tokens` 分组 |
| `token_mixer_large` | `token_dim`、`num_tokens`、`num_blocks`、`num_heads`、`hidden_factor`、`down_init_scale` | `num_heads` 必须满足 block 内部 reshape 约束 |
| `rankmixer` | `token_dim`、`num_tokens`、`num_blocks`、`num_heads`、`hidden_factor` | 当前实现要求 `num_heads == num_tokens` 以保持 residual shape |
| `rankup` | `token_dim`、`num_sparse_tokens`、`num_blocks`、`num_heads`、`permutation_seed`、`multi_embedding_tables`、`use_global_token`、`cross_token`、`num_task_tokens` | `num_sparse_tokens` 不能超过特征数；`cross_token.left/right` pooled dim 必须一致；task tower 可绑定 `task_i` |
| `hyformer` | `d`、`d_ff`、`num_queries`、`num_layers`、`hidden_factor` | `d` 必须能被 query boosting token 数整除；无序列特征时会退化为非序列 token memory |
| `uniformer` | `d`、`d_ff`、`num_layers`、`n_heads`、`num_tasks` | `d` 必须能被 `n_heads` 整除；`num_tasks` 决定可绑定的 `task_i` 表示数量 |

## 输入表示

| 输入方式 | 模型 | 说明 |
|---|---|---|
| `FeatureEmbeddings` concat | `lr`、`deepfm`、`mmoe`、`esmm`、`gdcn_esmm`、`pepnet` | 每个 feature 直接 embedding/pooling 后拼接 |
| 外部 `FeatureTokenizer` | `unimixer`、`token_mixer_large`、`rankmixer` | 构建模型时由训练/推理入口创建 tokenizer，权重通常带 `tokenizer.*` 前缀 |
| 内置 tokenizer | `rankup`、`hyformer`、`uniformer` | 模型内部自行管理 embedding、projection、token 构造 |

## 权重与上线注意事项

| 事项 | 要求 |
|---|---|
| Python/Rust 命名一致 | Python `state_dict` key 必须和 Rust `VarBuilder::pp()` 路径一致 |
| 结构参数变更 | embedding dim、token 数、query 数、tower hidden dims 等变更通常需要重新训练 |
| 输出语义 | `binary_logit` serving 转概率时需要 sigmoid；`probability/regression/score` 保持原值 |
| 新模型验证 | 至少跑 `cargo test --test model_smoke` 和 `python/tests/test_models.py` |
| 训练导出验证 | 导出后使用 `scale_rec_demo.verify_all` 做 Python/Rust 推理一致性检查 |
