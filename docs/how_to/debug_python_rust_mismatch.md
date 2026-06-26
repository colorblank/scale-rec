# Debug Python/Rust Mismatch

本文档说明 Python 训练导出和 Rust 推理输出不一致时的排查顺序。

## First check

先跑端到端验证：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

如果失败，按下面顺序缩小范围。

## 1. Check feature config

确认 Python 和 Rust 使用同一份 feature config：

```text
serving/configs/feature_config.yaml
```

如果通过 manifest 加载，服务会校验 sha256。裸 `.safetensors` 兼容模式下更容易传错 feature config。

## 2. Check model config

确认模型类型和 output_contract 一致：

```text
serving/configs/model_config.yaml
```

重点检查：

- `type`
- `params`
- `output_contract`
- task/output 名称

## 3. Check weight binding

Python `state_dict` key 必须匹配 Rust Candle `VarBuilder::pp()` 路径。

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.check_weight_bindings --models all
```

常见问题：

- Python module 名变化。
- Rust `vb.pp(...)` 路径变化。
- 新增层后未同步另一端。
- safetensors 里有缺失或 shape 不匹配 tensor。

## 4. Check preprocessing output

用同一行输入分别跑 Python DAG 和 Rust DAG，比较：

- DictMapper 输出。
- FeatureHash 输出。
- Split/List parser 输出。
- sequence padding/truncation。
- pooling 对应的 tensor shape。

中文和特殊符号场景重点检查：

- 文件解析是否已经列错位。
- NULL 标记是否一致。
- 分隔符和 escape/quote 规则是否一致。
- hash 输入字符串是否完全一致。

## 5. Check output kind

Rust serving 会根据 output kind 对 binary logit 做 sigmoid。Python 对比时也必须使用同样转换。

`scale_rec_demo.verify_all` 已经用 `serving_array()` 对齐该行为。

## 6. Reduce to one model and one row

先只验证一个模型：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_lr --force-train
```

LR 模型结构最简单，适合判断问题是在特征预处理还是复杂模型层。

## Related docs

- [Debug 与一致性验证](../tutorial/11_debug_and_consistency.md)
- [Model Config Reference](../reference/model_config.md)
- [Feature Config Reference](../reference/feature_config.md)
- [Rust Model Loading](../reference/rust_model_loading.md)
