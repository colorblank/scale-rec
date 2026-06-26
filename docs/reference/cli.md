# CLI Reference

本文档汇总训练、验证和服务启动常用命令。完整训练流程见 [Offline Training](../tutorials/offline_training.md)，多日训练见 [Train with Multi-day Files](../how_to/train_with_multi_day_files.md)。

所有命令从仓库根目录执行。

## Python entrypoints

| Entry point | Purpose |
|---|---|
| `python -m scale_rec_demo.generate_discover_data` | 生成 discover demo TSV |
| `python -m train.app.main discover` | 训练 discover TSV 模型 |
| `python -m train.app.main single` | 单文件通用训练入口 |
| `python -m train.app.main all` | 批量训练示例模型 |
| `python -m scale_rec_demo.verify_all` | Python 训练导出与 Rust 推理一致性验证 |
| `python -m scale_rec_demo.check_weight_bindings` | 检查 Python state_dict key 与 Rust Candle 路径绑定 |

## Generate demo data

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_discover_data \
  --label-policy examples/shared/discover_label_policy.yaml
```

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--label-policy` | `examples/shared/discover_label_policy.yaml` | demo 标签生成规则 |
| `--output` | `python/artifacts/demo/discover_train_data.txt` | 输出 TSV 路径 |

## Train discover model

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm \
  --run-version 20260526_120000
```

### Data arguments

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--data` | 空 | 单文件训练路径；未传 `--data-glob` 时使用 |
| `--eval-data` | 空 | 独立验证文件；格式、字段和顺序必须与训练文件一致 |
| `--data-glob` | 空 | 多日文件 glob；设置后优先于 `--data` |
| `--start-date` | 空 | `--data-glob` 闭区间开始日期，格式 `YYYYMMDD` |
| `--end-date` | 空 | `--data-glob` 闭区间结束日期，格式 `YYYYMMDD` |
| `--no-header` | false | TSV 无 header 时启用 |
| `--separator` | `\t` | 字段分隔符 |
| `--null-markers` | `NULL \N null None ""` | NULL 标记集合 |
| `--read-chunk-rows` | `0` | pandas `read_csv(chunksize=...)` 行数；0 表示自动 |
| `--fast-no-na` | false | 关闭 pandas NA 检测 |
| `--memory-map` | false | 对本地未压缩文件启用 pandas memory map |

### Config arguments

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--feature-config` | `examples/shared/feature_config_discover.yaml` | 训练和推理共享的特征 DAG 配置 |
| `--model-config` | 必填 | 模型结构与 output_contract 配置 |
| `--train-config` | 空 | 训练默认配置 YAML |
| `--init-weights` | 空 | 从 safetensors 初始化权重，不恢复 optimizer/scheduler |
| `--resume-from` | 空 | 从 checkpoint 恢复完整训练状态 |

### Artifact arguments

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--artifact-dir` | `python/artifacts/demo` | run 目录根路径 |
| `--publish-path` / `--export-path` | 自动生成 | 最终 serving 权重路径 |
| `--model-name` | 自动推导 | 逻辑模型名，写入 manifest |
| `--run-version` | 自动生成 | run/version 字符串 |
| `--keep-checkpoints` | `3` | 保留 checkpoint 数 |

## Verify all models

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--models` | `all` | 模型 key 列表，或 `all` |
| `--force-train` | false | 强制重新训练并覆盖 demo 权重 |
| `--threshold` | `1e-4` | Python/Rust 输出最大差异阈值 |

## Rust service

按 manifest 目录加载：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --worker-threads 4 \
  --blocking-threads 64
```

只加载单个 manifest：

```bash
cargo run --bin server --release -- \
  --model-path python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.manifest.yaml
```

更多加载规则见 [Rust Model Loading](rust_model_loading.md)。
