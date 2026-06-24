# 11. Debug 与一致性验证

[目录](README.md) | [上一章](10_performance_and_large_files.md)

这个仓库最容易出问题的地方不是模型能不能训，而是训练和推理是不是“同一件事”。

所以最后一章只讲四种验证手段：

1. 单样本 debug trace。
2. batch tensor 检查。
3. Python / Rust golden consistency。
4. 权重 key 排查。

## 单样本 trace

Python 侧 `FeatureDag(debug_mode=True)` 可以直接打印单样本的特征快照。

适合先看这几类问题：

- 原始字段有没有读到。
- 默认值有没有覆盖掉真值。
- 某个 operator 的输出是不是你预期的值。
- 最终 embeddable feature 有没有出错。

典型用法：

```python
import logging

from train.core.config import FlowConfig
from train.core.dag import FeatureDag

logging.basicConfig(level=logging.DEBUG)
fc = FlowConfig.from_yaml("examples/shared/feature_config_discover.yaml")
dag = FeatureDag(fc, debug_mode=True)
result = dag.execute({"user_id": 123, "item_id": 456})
print(result.features)
```

## 逐算子 tracer

如果你想看每个阶段的输入输出，用 `DebugTracer`。

它会记录：

- `DEFAULT_INIT`
- `RAW_OVERRIDE`
- `OPERATOR`

每个 sample 会形成一条 trace，里面包含每个阶段的输入、输出、异常和被覆盖字段。

这比只看最终结果更适合排查：

- list 解析失败。
- 空序列。
- NaN / Inf。
- 某个 operator 的上游输入类型不对。

## batch tensor 检查

`preprocess_batch()` 之后，模型看到的是 tensor，不再是单行字典。

排查 shape 问题时，直接打印 tensor：

```python
batch = [
    {"user_id": 123, "item_id": 456, "interest_keywords": "人工智能#0.9"},
    {"user_id": 124, "item_id": 457, "interest_keywords": "新能源#0.7"},
]
tensors = dag.preprocess_batch(batch)
for name, tensor in tensors.items():
    print(name, tuple(tensor.shape), tensor[:2])
```

这一步很适合确认：

- `seq_len` 是否对齐。
- `flatten` 是否把维度放大到预期大小。
- padding 是否在正确位置。

## golden consistency

最重要的一层验证是 Python 和 Rust 的 golden consistency。

仓库里已经有两套相关测试：

- Rust：`tests/golden_consistency.rs`
- Python：`python/tests/test_golden_consistency.py`

流程是：

1. 用同一份 feature config 和 sample fixture。
2. Python 和 Rust 分别执行 DAG。
3. 对比 feature 输出是否一致。

如果这里不一致，后面的模型输出基本不用看，先把特征层修对。

## 权重 key 排查

如果 Rust 加载权重失败，先不要怀疑数值精度，先看 key。

建议顺序：

1. Python 打印 `state_dict` keys。
2. 对照 Rust 端 `VarBuilder::pp()` 路径。
3. 看 manifest 里的 `weight_binding`。
4. 再看是否有额外 tokenizer / unimixer 前缀。

`python/src/train/app/export.py` 的 `print_state_dict_keys()` 就是专门给这一步用的。

## 常见一致性问题

- feature config 改了，但服务没同步发布。
- `embed_dim` 改了，但旧权重还在用。
- `tasks/task_config` 或 `output_contract` 变了，但模型、loss 或公开输出没同步。
- `flatten` / `mean` 的 pooling 语义在两端不一致。
- label 列被误当 feature 使用。

## 排查顺序建议

按这个顺序最省时间：

1. 先看 `docs/tutorial/02_samples_labels_tasks.md`，确认标签和任务契约。
2. 再看 `docs/tutorial/03_feature_contract.md`，确认 DAG 和 embedding。
3. 然后跑单样本 trace。
4. 再跑 batch tensor。
5. 最后做 Python / Rust golden 对比。

如果这五步都通过，说明系统主链路已经对齐了。
