#!/usr/bin/env bash
set -euo pipefail

platform="${PLATFORM:-linux/amd64}"
backend="${BACKEND:-default}"
tag=""
image_name="${IMAGE_NAME:-scale-rec-server}"
dockerfile="${DOCKERFILE:-docker/Dockerfile}"
context="${CONTEXT:-.}"
default_port="${DEFAULT_PORT:-8080}"
build_mode="load"
extra_args=()

usage() {
  cat <<'EOF'
Usage:
  docker/build.sh [--platform linux/amd64] [--backend default|cpu-mkl] [--tag TAG] [--default-port 8080] [--push|--load]

Environment overrides:
  PLATFORM      Build platform, default: linux/amd64
  BACKEND       Candle backend, default: default
  IMAGE_NAME    Base image name, default: scale-rec-server
  DOCKERFILE    Dockerfile path, default: docker/Dockerfile
  CONTEXT       Build context, default: .
  DEFAULT_PORT  Image default port, default: 8080
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --platform)
      platform="$2"
      shift 2
      ;;
    --backend)
      backend="$2"
      shift 2
      ;;
    --tag)
      tag="$2"
      shift 2
      ;;
    --image-name)
      image_name="$2"
      shift 2
      ;;
    --dockerfile)
      dockerfile="$2"
      shift 2
      ;;
    --context)
      context="$2"
      shift 2
      ;;
    --default-port)
      default_port="$2"
      shift 2
      ;;
    --push)
      build_mode="push"
      shift
      ;;
    --load)
      build_mode="load"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

case "$backend" in
  default)
    candle_features=""
    ;;
  cpu-mkl)
    candle_features="cpu-mkl"
    if [[ "$platform" != linux/amd64 ]]; then
      echo "cpu-mkl is only supported for linux/amd64 builds" >&2
      exit 1
    fi
    if [ "${DOCKERFILE:-}" = "docker/Dockerfile" ]; then
      dockerfile="docker/Dockerfile.mkl"
    fi
    ;;
  *)
    echo "unsupported backend: $backend" >&2
    exit 1
    ;;
esac

if [ -z "$tag" ]; then
  tag="${image_name}:${backend}-${platform//\//-}"
fi

build_args=(
  --platform "$platform"
  -f "$dockerfile"
  -t "$tag"
  --build-arg "CANDLE_FEATURES=$candle_features"
  --build-arg "DEFAULT_PORT=$default_port"
)

if [ "$build_mode" = "push" ]; then
  build_args+=(--push)
else
  build_args+=(--load)
fi

docker buildx build "${build_args[@]}" "${extra_args[@]}" "$context"
