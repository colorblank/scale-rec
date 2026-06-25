# 05. 多日训练与增量微调

[目录](README.md) | [上一章](04_offline_training_flow.md) | [下一章](06_model_structure_and_weight_binding.md)

生产训练通常不是一份单文件 TSV 跑完，而是按天读取多份数据，再在最近一天上做验证。

这一章只讲三件事：

1. `--data-glob` 如何解析成有序文件列表。
2. 多日训练时训练集和验证集如何切分。
3. `--init-weights` 和 `--resume-from` 的区别。

## 按日期读取文件

`python/src/train/app/cli.py` 支持这种写法：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 \
  --end-date 20260331 \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml
```

解析规则是：

- `--data-glob` 必须配 `--start-date` 和 `--end-date`。
- 日期范围是闭区间。
- 文件名里必须能提取出 `YYYYMMDD`。
- 同一天有多个文件时，会按文件名排序后依次读取。
- 缺失某一天文件会直接报错。

这套规则的目标是让训练输入顺序稳定可复现，而不是依赖 shell glob 的偶然顺序。

## 最后一天默认做验证集

`Trainer` 的多日读取逻辑会把最后一个日期文件单独留给验证：

```text
data_paths[:-1]  -> training
data_paths[-1]   -> eval
```

更细一点：

- 验证集从最后一天文件的前 `eval_samples` 个 batch 里采样。
- 训练时会跳过这部分 batch，避免训练和验证重叠。
- 如果只有一个文件，那它既承担训练样本，也会被切成 train/eval 两部分。

这意味着“最后一天质量不好”会直接影响监控指标，所以多日训练时要特别注意最后日期的标签分布。

## 使用独立验证文件

如果验证集已经单独产出，使用 `--eval-data`：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 \
  --end-date 20260331 \
  --eval-data data/eval_20260401.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --no-header
```

此时所有日期范围内的文件都用于训练，验证文件只用于评估。验证文件必须与训练文件使用相同的 header 规则、分隔符、字段数量、字段名称和字段顺序；流式训练仍由 `eval_samples` 控制加载的验证样本规模。

## 微调和断点恢复不是一回事

这两个参数经常被混用，但语义完全不同：

- `--init-weights`：只加载模型权重，重新开始训练，不恢复 optimizer / scheduler / epoch / EMA。
- `--resume-from`：从 checkpoint 恢复完整训练状态。

`python/src/train/app/main.py` 里也明确禁止两者同时使用。

### 适合用 `--init-weights` 的场景

- 想在旧模型上做新数据微调。
- 想切换 batch size 或训练超参，但不想带上旧优化器状态。
- 想把一个已发布权重当作初始化起点。

### 适合用 `--resume-from` 的场景

- 训练被中断，需要无缝接着跑。
- 想保留 EMA、loss weighting、scheduler、step 进度和随机数状态。
- 想继续同一条训练曲线，而不是“再训练一遍”。

## checkpoint 状态里保存了什么

恢复状态文件不仅有模型参数，还有这些内容：

- `optimizer_state`
- `loss_fn_state`
- `ema_state`
- `rng_state`
- `best_score`
- `stale_epochs`
- `best_epoch`
- `global_step`
- `batch_in_epoch`

所以真正的“resume”不是重新加载一个 `.safetensors`，而是把训练过程本身恢复回来。

## 什么时候应该用多日训练

适合以下场景：

- 日级数据持续增长。
- 你希望最后一天自然成为验证集。
- 你需要对旧模型做增量更新。

如果只是 demo 或小规模验证，单文件训练更简单，直接用 `--data` 即可。

## 训练前的检查

开始多日训练前，最好先确认：

1. 文件名中的日期和真实数据日期一致。
2. 日期范围里没有漏文件。
3. 最后一天的样本量足够做验证。
4. `--start-date` 不晚于 `--end-date`。
5. `--init-weights` 和 `--resume-from` 没同时传。

下一章讲模型结构和权重绑定。那部分的重点是：Python `state_dict` 的名字必须和 Rust `VarBuilder::pp()` 一一对应。
