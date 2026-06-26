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

详细设计见 [ADR 0001](../adr/0001-output-contract-v1.md)。

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
