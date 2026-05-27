# syntax=docker/dockerfile:1.7

# ---------- builder ----------------------------------------------------------
FROM rust:1.88-bookworm AS builder
WORKDIR /src

# Cache dependency builds: copy manifests first, fetch, then bring in sources.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
        --bin skeg --bin skeg-resp3 -p skeg-server && \
    cp target/release/skeg /usr/local/bin/skeg && \
    cp target/release/skeg-resp3 /usr/local/bin/skeg-resp3 && \
    strip /usr/local/bin/skeg /usr/local/bin/skeg-resp3

# ---------- runtime ----------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system --gid 1000 skeg && \
    useradd  --system --uid 1000 --gid 1000 --create-home --home /var/lib/skeg skeg && \
    mkdir -p /var/lib/skeg && \
    chown -R skeg:skeg /var/lib/skeg

COPY --from=builder /usr/local/bin/skeg        /usr/local/bin/skeg
COPY --from=builder /usr/local/bin/skeg-resp3  /usr/local/bin/skeg-resp3

USER skeg
WORKDIR /var/lib/skeg
VOLUME ["/var/lib/skeg"]

# Native protocol (used by skeg-client-rs, skeg-py, skeg-ollama) on 7379.
# RESP3 / Redis-compat on 6379 if user runs `--entrypoint skeg-resp3`.
EXPOSE 7379 6379

# Listen on all interfaces so the container is reachable from the host.
# Override with `-e SKEG_ADDR=...` for custom bind.
ENV SKEG_ADDR=0.0.0.0:7379 \
    SKEG_DATA_DIR=/var/lib/skeg \
    RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/skeg"]
