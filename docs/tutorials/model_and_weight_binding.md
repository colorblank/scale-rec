# Model Structure and Weight Binding

本教程介绍 Python 模型、Rust Candle 模型和 safetensors key 的绑定关系。

## Goal

理解为什么 Python `state_dict` key 必须和 Rust `VarBuilder::pp()` 路径一致。

## Model config

模型由 YAML 配置创建：

```yaml
type: gdcn_esmm
params:
  output_contract:
    version: 1
    ...
```

Python 和 Rust 都根据 model config 构建同名结构。

## Weight export

Python 训练后导出 safetensors：

```text
model.safetensors
```

Rust 用 Candle 加载这些 tensor。路径必须精确匹配。

## Binding examples

| Python module pattern | Rust path | Example key |
|---|---|---|
| `embeddings.emb_<name>.weight` | `embeddings.emb_<name>` | `embeddings.emb_user_id_idx.weight` |
| `hidden.0.weight` | `hidden.0` | `hidden.0.weight` |
| `output.<name>.weight` | `output.<name>` | `output.click_logit.weight` |

## Output contract

`output_contract` 统一定义：

- tower nodes。
- relation nodes。
- training objectives。
- evaluation metrics。
- public outputs。

模型应通过 output_contract 决定输出语义，而不是在代码里写死业务任务名。

## Verification

改模型结构后执行：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.check_weight_bindings --models all

PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

## Next

- 模型配置参考见 [Model Config Reference](../reference/model_config.md)。
- output_contract 设计见 [Model Config Reference](../reference/model_config.md#output_contract)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `scale_rec_demo.check_weight_bindings` | `--models all` checks every demo model | [CLI Reference: Check weight bindings](../reference/cli.md#check-weight-bindings) |
| `scale_rec_demo.verify_all` | `--models all --force-train` retrains and compares Python/Rust outputs | [CLI Reference: Verify all](../reference/cli.md#verify-all) |
