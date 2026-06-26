# Getting Started

本页给出一条最短可运行路径：生成 demo 数据，训练并导出模型，启动 Rust HTTP 服务，并验证 Python 与 Rust 推理一致性。所有命令从仓库根目录执行。

## Prerequisites

- Rust toolchain，可运行 `cargo`。
- Python 依赖通过 `uv` 管理，不需要手动激活 virtualenv。
- 本地命令统一带上 `PYTHONPATH=python/src:$PYTHONPATH`。

常用检查：

```bash
cargo check
PYTHONPATH=python/src:$PYTHONPATH uv run --project python pytest python/tests/ -q
```

## Generate demo data

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_demo_data \
  --label-policy examples/shared/demo_label_policy.yaml
```

输出：

```text
python/artifacts/demo/demo_train_data.txt
```

demo 数据是无 header 的 TSV，列顺序来自 `examples/shared/feature_config_demo.yaml`。

## Train a model

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data python/artifacts/demo/demo_train_data.txt \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name demo_train \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm \
  --run-version 20260526_120000
```

训练完成后，serving 目录包含推理侧需要的文件：

```text
python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/
├── model.manifest.yaml
├── model.safetensors
└── configs/
    ├── feature_config.yaml
    └── model_config.yaml
```

推荐服务端加载 `model.manifest.yaml` 或扫描包含 manifest 的 `model-dir`。

## Train with independent eval data

如果验证集来自独立文件，传入 `--eval-data`：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main demo \
  --data data/train.tsv \
  --eval-data data/eval.tsv \
  --feature-config examples/shared/feature_config_demo.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --run-name demo_train \
  --no-header
```

验证文件必须与训练文件格式一致、字段一致、字段顺序一致。传入 `--eval-data` 后，训练文件不再切出验证样本。

## Serve the model

按目录加载所有 serving manifest：

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

检查服务：

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm/features
```

Pointwise 推理：

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","features":[{"user_id":42,"item_id":500}]}'
```

Broadcast 推理：

```bash
curl -X POST http://127.0.0.1:8080/predict/broadcast \
  -H 'Content-Type: application/json' \
  -d '{"model":"model_gdcn_esmm","user":{"user_id":42},"items":[{"item_id":500},{"item_id":501}]}'
```

完整 HTTP 协议见 [HTTP API](reference/http_api.md)。

## Verify Python and Rust consistency

端到端验证会执行 Python 训练、safetensors 导出、Rust `demo_inference` 推理和输出比对：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models all --force-train
```

也可以只验证部分模型：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all \
  --models demo_lr,demo_gdcn_esmm,demo_rankmixer \
  --force-train
```

期望输出：

```text
Overall Consistency Status: PASS
```

## Next steps

- 想理解完整链路：读 [Tutorials](tutorials/index.md)。
- 想查训练参数和 artifact：读 [CLI Reference](reference/cli.md) 和 [Artifact Reference](reference/artifacts.md)。
- 想查 HTTP 请求响应：读 [HTTP API](reference/http_api.md)。
- 想查特征算子：读 [Feature operators](reference/feature_operators.md)。

## Command arguments

| Command | Arguments used here | Full parameter table |
|---|---|---|
| `cargo check` | No project-specific flags | [Development Reference](reference/development.md) |
| `pytest` | `python/tests/ -q` runs the Python test suite quietly | [Development Reference](reference/development.md) |
| `scale_rec_demo.generate_demo_data` | `--label-policy` selects the demo label policy YAML | [CLI Reference: Generate demo data](reference/cli.md#generate-demo-data) |
| `train.app.main demo` | Training data, config, artifact, runtime and TSV reader flags | [CLI Reference: Train demo](reference/cli.md#train-demo) |
| `cargo run --bin server` | `--model-dir` and `--model-path` control model loading | [CLI Reference: Rust server](reference/cli.md#rust-server) |
| `curl /health` / `/models` / `/predict` / `/predict/broadcast` | HTTP method, endpoint path, JSON body and `Content-Type` header | [HTTP API](reference/http_api.md) |
| `scale_rec_demo.verify_all` | `--models`, `--force-train`, `--threshold` | [CLI Reference: Verify all](reference/cli.md#verify-all) |
