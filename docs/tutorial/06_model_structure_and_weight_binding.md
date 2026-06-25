# 06. 模型结构与权重绑定

[目录](README.md) | [上一章](05_multi_day_incremental.md) | [下一章](07_evaluation_and_feature_quality.md)

训练能跑起来不够，权重还必须能被 Rust 端直接加载。

这章只讲一个关键原则：

```text
模型结构由 Python 训练侧定义，Rust 推理侧必须用同名层级和同样的权重 key 去加载。
```

## 模型不是从 config 里“自动推断”出来的

`python/src/train/models/__init__.py` 维护一个模型注册表。`ModelConfig.build()` 不是 case-by-case 的大 `if`，而是通过 registry 构建模型。

当前仓库里已经注册的模型包括：

- `lr`
- `deepfm`
- `mmoe`
- `esmm`
- `gdcn_esmm`
- `unimixer`
- `token_mixer_large`
- `rankmixer`

这些模型分成两类：

- 传统打分模型：`lr`、`deepfm`、`mmoe`。
- 多任务排序模型：`esmm`、`gdcn_esmm`、`unimixer`、`token_mixer_large`、`rankmixer`。

## 输出契约

全部 8 个示例模型使用 `output_contract.version: 1`：

- `graph.towers` 构建标量任务塔，塔输出只允许 `binary_logit`、`regression`、`score`。
- `graph.relations` 执行 `sigmoid/multiply/add/identity` 类型化 DAG。
- `objectives` 由 `ObjectiveEngine` 读取内部节点计算 loss。
- `metrics` 独立声明评估节点和标签。
- `outputs` 将内部节点映射为稳定的公开输出。

Python 和 Rust 都使用 `OutputHead` 执行相同的塔和关系图。完整前向结果是
`ModelExecution`：

```text
ModelExecution.nodes   = 所有 tower/relation 内部节点
ModelExecution.outputs = output_contract.outputs 指定的公开输出
```

普通 `forward()` 只返回公开输出；训练和评估通过 `forward_execution()` 使用内部节点。
legacy 模型的默认实现会把同一份 `ModelOutput` 同时作为 nodes 和 outputs。
legacy 配置仍可加载，但不能与 `output_contract` 混用。

## ESMM / GDCN+ESMM 的关系输出

`gdcn_esmm.yaml` 等多任务配置使用 `graph.relations` 描述派生任务：

```yaml
relations:
  - {name: click_prob, op: sigmoid, inputs: [click_logit]}
  - {name: cvr_prob, op: sigmoid, inputs: [cvr_logit]}
  - {name: ctcvr_prob, op: multiply, inputs: [click_prob, cvr_prob]}
```

`objectives` 可以直接引用联合概率。例如 `ctcvr_prob` 在两个 logit
分别 sigmoid 后相乘，并用概率版 BCE 参与训练。概率关系的计算顺序和 loss 类型都由
契约校验，不再依赖任务名约定。

LR、DeepFM、ESMM、GDCN-ESMM 和三类 mixer 向 `OutputHead` 提供 `shared` 表示。MMoE
按 `graph.towers[].input` 声明构建 gate，并将 `click_rep` 等命名表示交给对应塔。

## UniMixer / TokenMixer / RankMixer 的特殊点

这三类模型都需要外部 `FeatureTokenizer`：

- `unimixer`
- `token_mixer_large`
- `rankmixer`

它们的输入不是简单的 embedding concat，而是先把离散特征打包成 token，再做 mixer / attention / block 结构运算。

因此它们的构建逻辑会多一个 tokenizer 层级，导出权重时也要注意前缀对齐。

## 为什么权重 key 会失配

Python 训练侧导出的 `state_dict` key，必须和 Rust 端 `VarBuilder::pp()` 的路径一致。

常见模式如下：

| Python 模块写法 | Rust 路径 | 示例 key |
|---|---|---|
| `setattr(self, f"emb_{name}", nn.Embedding)` | `vb.pp(format!("emb_{}", name))` | `embeddings.emb_user_id.weight` |
| `self.hidden = nn.ModuleDict({"0": Linear})` | `vb.pp("hidden.0")` | `hidden.0.weight` |
| `self.output = nn.Linear(...)` | `vb.pp("output")` | `output.weight` |
| `self.output = nn.ModuleDict({str(n): Linear})` | `vb.pp(format!("output.{}", n))` | `output.1.weight` |
| `nn.Parameter(torch.zeros(1))` | `vb.get_with_hints(..., Const(0.0))` | `global_bias` |
| `self.output_head.towers[name]` | `vb.pp("output_head").pp("towers").pp(name)` | `output_head.towers.click_logit.output.2.weight` |

`python/src/train/app/export.py` 里提供了 `print_state_dict_keys(model)`，就是给你排查这类问题用的。

## 什么时候旧权重还能用

一般规律是：

- 改训练超参，不改结构，旧权重通常还能加载。
- 改 embedding 维度、vocab size、token 数、task 数，旧权重通常不能 strict 加载。
- 改 feature config 里会进入模型的 embeddable feature，旧权重基本就需要重训。

所以只要你改了会影响输入维度的东西，就不要假设历史 safetensors 还能直接复用。

## 如何检查绑定是否正确

最直接的检查顺序是：

1. 在 Python 里打印 `state_dict` keys。
2. 对照 Rust 中的 `VarBuilder::pp()` 路径。
3. 跑 `cargo test --test model_smoke`。
4. 再跑 `python -m scale_rec_demo.verify_all`。

如果两边输出的 logits 不一致，先查 key，再查 feature config，再查模型结构，最后才查数值误差。

下一章讲评估和 feature quality。那部分会把“训练是不是有效”拆成指标、质量和可观测性三层。
