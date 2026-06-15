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

## 任务定义是模型和 loss 的共同契约

`examples/models/*.yaml` 里最重要的是 `tasks:`：

```yaml
tasks:
  - {name: click, label: is_click, loss: bce, metrics: [auc, logloss]}
  - {name: stay, label: stay_time_label, loss: weighted_bce_stay, metrics: [mae, mse]}
```

含义是：

- `name` 是模型输出名。
- `label` 是 batch 里的监督列。
- `loss` 是这个任务的损失函数。
- `metrics` 是评估时要记录的指标。

`MultiTaskLoss` 会强制检查：

- 模型是否少输出了某个 task。
- 模型是否多输出了配置里没定义的非 `ct*` task。
- batch 里是否存在对应 label 列。

## ESMM / GDCN+ESMM 的关系输出

`gdcn_esmm.yaml` 和 `esmm.yaml` 使用 `task_config.relations` 描述派生任务：

```yaml
relations:
  - {target: ctcvr, sources: [click, cvr], op: multiply}
  - {target: ctdetail, sources: [click, detail], op: multiply}
  - {target: ctstock, sources: [click, stock], op: multiply}
  - {target: ctstay, sources: [detail, stay], op: multiply}
```

这类输出不是额外监督目标，而是关系概率。训练 loss 主要还是落在基础 task 上。

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
