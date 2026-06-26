# 开发环境

本文档描述本地开发、测试和验证命令。所有命令默认从仓库根目录执行。

## 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| Rust | stable，edition 2021 | 由 `Cargo.toml` 管理依赖 |
| Python | `>=3.10` | `python/pyproject.toml` 声明；当前 `python/.python-version` 为 `3.14` |
| uv | 需要 | Python 依赖、脚本和测试统一通过 uv 运行 |
| uvx | 需要 | 运行 ruff 等独立工具 |

Rust 依赖会由 Cargo 自动下载和编译。Python 项目文件位于 `python/`，因此建议使用 `uv run --project python ...`，并显式设置 `PYTHONPATH=python/src:$PYTHONPATH`。

## 初始化检查

确认工具链：

```bash
rustc --version
cargo --version
uv --version
uvx --version
```

同步 Python 依赖：

```bash
uv sync --project python
```

如果只是运行命令，`uv run --project python ...` 也会按需创建和使用虚拟环境。

## Rust 开发

常用检查：

```bash
cargo fmt
cargo check
cargo test
```

只跑服务端相关单元测试：

```bash
cargo test server:: --lib
```

只跑模型 smoke 测试：

```bash
cargo test --test model_smoke
```

构建服务和压测工具：

```bash
cargo build --release --bin server --bin bench
```

按后端构建：

```bash
# macOS Accelerate
cargo build --release --features macos-accelerate --bin server --bin bench

# macOS Metal
cargo build --release --features macos-metal --bin server --bin bench

# Linux MKL
RUSTFLAGS="-C target-cpu=native" \
cargo build --release --features cpu-mkl --bin server --bin bench
```

启动服务：

```bash
cargo run --bin server --release -- \
  --model-dir python/artifacts/demo \
  --port 8080
```

## Python 开发

Python 包名为 `train`，源码在 `python/src/train`。运行模块时使用：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main --help
```

生成 discover demo 数据：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.generate_discover_data \
  --label-policy examples/shared/discover_label_policy.yaml
```

训练 demo 模型：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data python/artifacts/demo/discover_train_data.txt \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --train-config examples/shared/train_defaults.yaml \
  --epochs 10 --batch-size 128 --no-header --eval-samples 400 \
  --artifact-dir python/artifacts/demo \
  --model-name model_gdcn_esmm
```

多日文件和增量微调：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m train.app.main discover \
  --data-glob 'data/user_*.txt' \
  --start-date 20260325 --end-date 20260331 \
  --feature-config examples/shared/feature_config_discover.yaml \
  --model-config examples/models/gdcn_esmm.yaml \
  --init-weights python/artifacts/demo/model_gdcn_esmm/20260526_120000/serving/model.safetensors \
  --epochs 3 --batch-size 1024 --no-header
```

这里的配置分层和 [训练手册](../TRAINING_GUIDE.md#训练流程) 保持一致：`feature_config`
管特征编排，`label_policy` 只管 demo 标签，`model_config` 通过原生
`output_contract` 定义模型输出语义，`train_config` 定义训练默认值和评估策略。legacy
字段仅用于兼容旧配置。

调试特征预处理时，先用单样本 `FeatureDag(debug_mode=True).execute(row)` 看 source/default/operator 输出，再用 `dag.preprocess_batch(rows)` 看最终 tensor shape/value；训练侧整体质量看 run manifest 里的 `feature_quality.*` 指标。详细步骤见 [训练手册 - 特征预处理 Debug](../TRAINING_GUIDE.md#特征预处理-debug)。

## Python 测试与格式化

运行 Python 测试：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  pytest python/tests/ -v
```

ruff 检查：

```bash
uvx --offline ruff check python/src/
```

ruff 格式化：

```bash
uvx --offline ruff format python/src/
```

mypy 配置位于 `python/pyproject.toml`。需要类型检查时：

```bash
uv run --project python mypy
```

## 端到端验证

端到端训练-导出-推理验证会串起训练、权重导出、PyTorch 推理、Rust 推理和结果比对：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_lr,discover_gdcn_esmm,discover_unimixer,discover_token_mixer_large,discover_rankmixer --force-train
```

验证全部 demo 主线：

```bash
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all
```

最近一次全链路验证已覆盖 `discover_lr`、`discover_gdcn_esmm`、`discover_unimixer`、`discover_token_mixer_large`、`discover_rankmixer`，Python 训练、safetensors 导出、Rust 推理和输出比对均通过；Rust 侧的 `cargo test --test model_smoke` 也同步通过。

如果本地线程环境对 OpenMP/MKL 比较敏感，可以先设置：

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
KMP_INIT_AT_FORK=FALSE
```

## 常见开发流程

修改 Rust 推理逻辑后：

```bash
cargo fmt
cargo check
cargo test server:: --lib
cargo test
```

修改 Python 训练或 manifest 逻辑后：

```bash
uvx --offline ruff check python/src/
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  pytest python/tests/ -v
```

修改双端模型或权重命名后：

```bash
cargo test --test model_smoke
PYTHONPATH=python/src:$PYTHONPATH uv run --project python \
  python -m scale_rec_demo.verify_all --models discover_gdcn_esmm --force-train
```

## 注意事项

- 不要手动激活虚拟环境；使用 `uv run --project python`。
- 从仓库根目录执行命令，路径和文档示例都按根目录编写。
- Python 运行时保持 `PYTHONPATH=python/src:$PYTHONPATH`，否则 `train` 和 `scale_rec_demo` 模块可能无法解析。
- Rust `server` 默认监听 `0.0.0.0:8080`；本地调用使用 `http://127.0.0.1:8080`。
- Linux MKL 压测建议设置 `MKL_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1`，避免每个请求内部再开过多 BLAS 线程。
