# Model Config Reference

model config YAML 定义模型结构、任务图、训练目标、评估指标和公开输出。当前示例模型统一使用 `output_contract.version: 1`，legacy task 字段仅保留兼容。

## Example

```yaml
type: gdcn_esmm

params:
  embed_dim: 16
  hidden_dims: [128, 64]

  output_contract:
    version: 1
    graph:
      towers:
        - name: click_logit
          kind: binary_logit
          input: shared
          hidden_dims: [64, 32]
      relations:
        - name: ctr
          op: sigmoid
          inputs: [click_logit]
    objectives:
      - name: click_loss
        source: click_logit
        label: is_click
        weight: 1.0
        loss:
          type: binary_cross_entropy_with_logits
    metrics:
      - name: click_auc
        source: click_logit
        label: is_click
        type: auc
    outputs:
      - name: ctr
        source: ctr
```

## Registered model types

| `type` | 说明 |
|---|---|
| `lr` | 线性 baseline |
| `deepfm` | FM + deep network |
| `mmoe` | Multi-gate Mixture-of-Experts |
| `esmm` | ESMM 多任务结构 |
| `gdcn_esmm` | Gated Deep & Cross + ESMM |
| `unimixer` | token mixer 模型 |
| `token_mixer_large` | larger token mixer |
| `rankmixer` | ranking mixer 模型 |

示例配置在 `examples/models/`。

## output_contract

`output_contract` 是模型输出、训练目标、评估指标和公开输出的统一契约。

| 区域 | 说明 |
|---|---|
| `graph.towers` | 直接由模型 shared representation 产生的 tower 节点 |
| `graph.relations` | 基于 tower 或 relation 的派生节点，例如 sigmoid/multiply/add |
| `objectives` | 训练 loss 列表 |
| `metrics` | 评估指标列表 |
| `outputs` | 对外公开输出列表 |

引入它的原因是消除旧模型里的隐式约定：旧路径把任务塔、ESMM 概率关系、公开输出和
loss 规则分散在模型实现、训练代码和 serving 适配逻辑中。同一个任务名在不同模型里
可能代表 logit，也可能代表 probability，容易导致 loss 选择、指标计算和
Rust/Python 一致性依赖约定而不是配置。

`output_contract.version: 1` 把这些语义全部写进配置：

- `graph.towers` 从 backbone 的命名表示构建标量任务塔。
- `graph.relations` 在 tower 输出之上构建无参数、有类型的关系 DAG。
- `objectives` 声明训练目标、标签、损失、权重和可选样本 mask。
- `metrics` 独立声明评估节点、标签、指标和可选样本 mask。
- `outputs` 将内部节点投影为稳定的公开输出名称。

### Node types and relations

Tower 只允许输出三类节点：

| `kind` | 语义 |
|---|---|
| `binary_logit` | 二分类 logit，适用于 BCE-with-logits |
| `regression` | 回归值 |
| `score` | 排序分或通用连续分数 |

概率必须通过显式 `sigmoid` relation 产生。`multiply` 只接受两个及以上 probability
输入，`add` 只接受两个及以上 regression 输入，`identity` 保留输入类型。relation 按
DAG 拓扑执行；循环、未知引用、重复名称和未消费节点都会在构建前拒绝。

### Loss and metric type checks

loss 与节点类型严格匹配：

| Loss | 允许的 source 类型 |
|---|---|
| `binary_cross_entropy_with_logits` | `binary_logit` |
| `weighted_bce_stay` | `binary_logit` |
| `binary_cross_entropy` | `probability` |
| `mse` / `mae` / `huber` | `regression` 或 `score` |

`binary_cross_entropy` 会对 probability 使用显式 epsilon 截断。

metric 也按节点类型处理：

- `auc` 接受 logit 或 probability；logit 会在 metric 入口转换为 probability，probability 不会重复 sigmoid。
- `logloss` 只接受 probability。
- `mae` / `mse` 只接受 regression 或 score。

serving 只序列化 `outputs` 指定的节点，不再自动执行 sigmoid 或任务名映射。

### Consistency and validation

Rust 和 Python 解析并校验同一份 schema。两端共享接受/拒绝 fixtures，规范化过程会展开
默认值、稳定排序节点，并以统一浮点表示生成 canonical JSON，供 manifest 保存完整契约及摘要。

训练侧还会把 `output_contract` 与 feature config 联合校验：

- objective / metric 引用的 label 必须是 `role: label`。
- mask source 必须存在。
- 原生契约引用的 label 不允许引用 feature/discard source。

当前仍保留 `FlowConfig` 的兼容格式；“label 不配置默认值”尚未在实际训练配置中强制。

### Compatibility and migration

原生 `output_contract` 与旧 `tasks` / `task_config` / `label_col_map` / `metrics` 禁止混用。
仓库中的模型示例均使用原生 `output_contract`；旧字段只作为兼容路径保留，新配置不应继续使用。

当前状态：

- 8 个注册模型 `lr`、`deepfm`、`mmoe`、`esmm`、`gdcn_esmm`、`unimixer`、
  `token_mixer_large`、`rankmixer` 均支持原生契约。
- shared-backbone 模型向 `OutputHead` 暴露 `shared` 表示。
- MMoE 按 `graph.towers[].input` 的首次出现顺序构建 gate，并暴露同名表示。
- contract 模型的 `forward()` 只返回公开输出，`forward_execution()` 同时保留内部节点，
  供训练和评估使用。
- Python 导出与 Rust 加载已覆盖全部 8 个示例模型的 key/shape 绑定检查。
- manifest 保存规范化契约仍属于后续发布治理工作。

## Loss weighting

启用 `output_contract` 时，训练只支持 static loss weighting。每个 objective 的权重来自：

```yaml
objectives:
  - name: cvr_loss
    weight: 2.0
```

总损失为：

```text
total_loss = sum(objective_loss * objective.weight)
```

legacy `tasks` 路径支持 `static`、`equal` 和 `uncertainty`，但新配置不应继续使用 legacy 字段。

## Weight binding

Python 导出的 `state_dict` key 必须和 Rust Candle `VarBuilder::pp()` 路径一致。训练导出时会写入 manifest 的 `weight_binding`，服务加载时根据 binding 校验 key 和 shape。

修改模型结构后必须跑：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `scale_rec_demo.verify_all` | `--models all --force-train` retrains and verifies all demo models | [CLI Reference: Verify all](cli.md#verify-all) |
