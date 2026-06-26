# End-to-end Recommendation Pipeline

本教程建立 scale-rec 的全链路心智模型：从样本表、特征 DAG、PyTorch 训练，到 safetensors 导出和 Rust HTTP 推理。

## Goal

理解一次完整推荐排序请求在离线和在线两侧如何流动：

```text
样本 TSV
  -> Python feature DAG
  -> PyTorch model
  -> safetensors + serving manifest
  -> Rust feature DAG
  -> Candle model
  -> HTTP response
```

## Prerequisites

- 已读 [Getting Started](../getting_started.md)。
- 能从仓库根目录运行 `cargo` 和 `uv` 命令。

## Key files

| Area | Files |
|---|---|
| Feature contract | `examples/shared/feature_config_demo.yaml` |
| Model config | `examples/models/*.yaml` |
| Training entrypoint | `python/src/train/app/main.py` |
| Demo verification | `python/src/scale_rec_demo/verify_all.py` |
| Rust serving | `src/server/`、`src/bin/server.rs` |

## Steps

1. 生成 demo 数据：

   ```bash
   PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
     python -m scale_rec_demo.generate_demo_data \
     --label-policy examples/shared/demo_label_policy.yaml
   ```

2. 训练并发布模型：

   ```bash
   PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
     python -m train.app.main demo \
     --data python/artifacts/demo/demo_train_data.txt \
     --feature-config examples/shared/feature_config_demo.yaml \
     --model-config examples/models/gdcn_esmm.yaml \
     --train-config examples/shared/train_defaults.yaml \
     --epochs 1 --batch-size 128 --no-header \
     --artifact-dir python/artifacts/demo \
     --model-name model_gdcn_esmm
   ```

3. 验证 Python/Rust 一致：

   ```bash
   PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
     python -m scale_rec_demo.verify_all --models demo_gdcn_esmm --force-train
   ```

4. 启动 HTTP 服务：

   ```bash
   cargo run --bin server --release -- \
     --model-dir python/artifacts/demo
   ```

## Checkpoint

端到端验证通过时会输出：

```text
Overall Consistency Status: PASS
```

## Next

- 样本和标签见 [Samples, Labels, and Tasks](samples_labels_and_tasks.md)。
- 特征 DAG 见 [Feature DAG](feature_dag.md)。
- 发布产物见 [Artifact Publishing](artifact_publishing.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `scale_rec_demo.generate_demo_data` | `--label-policy` selects demo label policy YAML | [CLI Reference: Generate demo data](../reference/cli.md#generate-demo-data) |
| `train.app.main demo` | Data, feature/model configs, training and artifact flags | [CLI Reference: Train demo](../reference/cli.md#train-demo) |
| `scale_rec_demo.verify_all` | `--models` selects model keys; `--force-train` retrains before comparison | [CLI Reference: Verify all](../reference/cli.md#verify-all) |
| `cargo run --bin server` | `--model-dir` scans serving artifacts | [CLI Reference: Rust server](../reference/cli.md#rust-server) |
