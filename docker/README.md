# Rust 服务 Docker 打包

这里提供 Rust HTTP 服务的 Docker 打包入口。当前方案只覆盖 Linux 容器运行时；`macos-accelerate` 和 `macos-metal` 仍然是本机 macOS 构建，不走 Docker。

## 支持矩阵

| 运行环境 | Candle 后端 | 说明 |
| --- | --- | --- |
| `linux/amd64` | `default` | 纯 CPU，通用基线 |
| `linux/amd64` | `cpu-mkl` | x86_64 / MKL 专用变体 |
| `linux/arm64` | `default` | 纯 CPU，适合 ARM 机器 |

不支持的组合：

- `cpu-mkl` + `linux/arm64`
- `macos-accelerate`
- `macos-metal`

这些后端需要在 macOS 上本机构建：

```bash
cargo build --release --bin server --features macos-accelerate
cargo build --release --bin server --features macos-metal
```

## 构建

默认构建 Linux amd64 的通用 CPU 镜像：

```bash
./docker/build.sh --load
```

构建 MKL 版本：

```bash
./docker/build.sh --backend cpu-mkl --platform linux/amd64 --load
```

MKL 版本使用单独的 `docker/Dockerfile.mkl`，默认会走 `cpu-mkl` 编译特征，避免和通用 CPU 版混在一起。

构建 ARM64 版本：

```bash
./docker/build.sh --platform linux/arm64 --load
```

如果要推送到镜像仓库，用 `--push` 替换 `--load`。

## 运行

容器默认通过环境变量启动服务：

- `MODEL_DIR`：批量模型目录，默认 `/models`。未设置 `MODEL_PATH` 时，服务会扫描这里的 serving manifest
- `MODEL_PATH`：显式模型路径，支持 serving manifest、目录或旧 `.safetensors`；多个路径用英文逗号分隔。设置后优先于 `MODEL_DIR`
- `FEATURE_CONFIG`：可选，只在加载无 serving manifest 的旧 `.safetensors` 产物时作为 feature config fallback
- `PORT`：监听端口，默认 `8080`
- `WORKER_THREADS`：Tokio worker 线程数，可选
- `BLOCKING_THREADS`：Tokio blocking 线程数，可选

推荐把训练发布产物目录挂载到 `/models`，让服务自动扫描 `*.manifest.yaml`、`*_manifest.yaml` 或 `model_manifest.yaml`。manifest 会指定权重、模型配置、特征配置、sha256 和 `weight_binding`，容器不需要额外挂载 `FEATURE_CONFIG`。

示例：

```bash
docker run --rm \
  -p 8080:8080 \
  -e MODEL_DIR=/models \
  -v "$PWD/models:/models:ro" \
  scale-rec-server:default-linux-amd64
```

只加载单个 manifest：

```bash
docker run --rm \
  -p 8080:8080 \
  -e MODEL_PATH=/models/model_gdcn_esmm.manifest.yaml \
  -v "$PWD/models:/models:ro" \
  scale-rec-server:default-linux-amd64
```

加载多个显式路径：

```bash
docker run --rm \
  -p 8080:8080 \
  -e MODEL_PATH=/models/ranker_v1/model_manifest.yaml,/models/ranker_v2/model_manifest.yaml \
  -v "$PWD/models:/models:ro" \
  scale-rec-server:default-linux-amd64
```

旧产物没有 manifest 时，可以显式提供 `.safetensors` 和 fallback feature config：

```bash
docker run --rm \
  -p 8080:8080 \
  -e MODEL_PATH=/models/model_gdcn_esmm.safetensors \
  -e FEATURE_CONFIG=/config/feature_config.yaml \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/examples/feature_config_discover.yaml:/config/feature_config.yaml:ro" \
  scale-rec-server:default-linux-amd64
```

查询已加载模型和版本：

```bash
curl http://127.0.0.1:8080/models
curl http://127.0.0.1:8080/models/model_gdcn_esmm
```

如果你想覆盖默认启动命令，可以直接给容器传入命令，`entrypoint.sh` 会原样执行：

```bash
docker run --rm scale-rec-server:default-linux-amd64 \
  /usr/local/bin/scale-rec-server --help
```

## 说明

Dockerfile 是多阶段构建：

1. builder 阶段用 `cargo build --release --bin server`
2. runtime 阶段只保留 `server` 二进制和运行时依赖

MKL 变体不是靠基础镜像内置 MKL 包实现的，而是靠 Candle 的 `cpu-mkl` 编译特征把对应后端编进二进制。

这套包装不把模型和特征配置打进镜像，默认通过挂载卷提供，便于同一个镜像在不同模型目录上复用。
