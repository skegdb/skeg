# syntax=docker/dockerfile:1.7

# ---------- builder ----------------------------------------------------------
# RUST_VERSION defaults to the workspace MSRV. The AVX-512 kernels need 1.89,
# so the avx512 image passes --build-arg RUST_VERSION=1.89.
ARG RUST_VERSION=1.88
FROM rust:${RUST_VERSION}-bookworm AS builder
WORKDIR /src

# Empty for the default image. Set to `avx512` to compile the AVX-512 kernels
# in as well; they still select themselves only after a runtime CPU check, so
# the resulting binary runs on any x86_64 machine.
ARG CARGO_FEATURES=

# Cache dependency builds: copy manifests first, fetch, then bring in sources.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
        --bin skeg --bin skeg-resp3 -p skeg-server \
        ${CARGO_FEATURES:+--features $CARGO_FEATURES} && \
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

# Docker's default is already SIGTERM; spelling it here makes the image's
# durable shutdown contract part of its metadata rather than an assumption.
STOPSIGNAL SIGTERM

# Listen on all interfaces so the container is reachable from the host via
# `-p`. This image has no authentication, so the server refuses to start
# unless the operator explicitly opts in with
# `-e SKEG_ALLOW_UNAUTHENTICATED_NETWORK=1` on the `docker run` command -
# see the README quickstart. Override SKEG_ADDR for a custom bind. The
# operator is relied on to publish the port on the host loopback only
# (`-p 127.0.0.1:7379:7379`) or otherwise keep the container network-isolated.
ENV SKEG_ADDR=0.0.0.0:7379 \
    SKEG_DATA_DIR=/var/lib/skeg \
    RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/skeg"]
