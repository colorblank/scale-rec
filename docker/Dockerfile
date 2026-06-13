# syntax=docker/dockerfile:1.7

ARG RUST_IMAGE=rust:1.89-bookworm
ARG DEFAULT_PORT=8080

FROM ${RUST_IMAGE} AS builder

ARG CANDLE_FEATURES=""

ENV CARGO_NET_GIT_FETCH_WITH_CLI=true \
    RUST_BACKTRACE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libssl-dev \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    set -eux; \
    if [ -n "${CANDLE_FEATURES}" ]; then \
      cargo build --release --locked --bin server --features "${CANDLE_FEATURES}"; \
    else \
      cargo build --release --locked --bin server; \
    fi

FROM debian:bookworm-slim AS runtime

ARG DEFAULT_PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/server /usr/local/bin/scale-rec-server
COPY docker/entrypoint.sh /usr/local/bin/scale-rec-entrypoint

RUN chmod +x /usr/local/bin/scale-rec-entrypoint

ENV MODEL_DIR=/models \
    PORT=${DEFAULT_PORT}

EXPOSE ${DEFAULT_PORT}

ENTRYPOINT ["/usr/local/bin/scale-rec-entrypoint"]
