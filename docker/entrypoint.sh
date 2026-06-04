#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

cmd=(
  "${SERVER_BIN:-/usr/local/bin/scale-rec-server}"
  --port "${PORT:-8080}"
)

if [ -n "${MODEL_PATH:-}" ]; then
  IFS=',' read -r -a model_paths <<< "${MODEL_PATH}"
  for model_path in "${model_paths[@]}"; do
    if [ -n "${model_path}" ]; then
      cmd+=(--model-path "${model_path}")
    fi
  done
else
  cmd+=(--model-dir "${MODEL_DIR:-/models}")
fi

if [ -n "${FEATURE_CONFIG:-}" ]; then
  cmd+=(--feature-config "${FEATURE_CONFIG}")
fi

if [ -n "${WORKER_THREADS:-}" ]; then
  cmd+=(--worker-threads "${WORKER_THREADS}")
fi

if [ -n "${BLOCKING_THREADS:-}" ]; then
  cmd+=(--blocking-threads "${BLOCKING_THREADS}")
fi

exec "${cmd[@]}"
