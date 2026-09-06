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

# Build from the restricted, locked server graph. The root workspace's vendored
# SDK adapters are not dependencies of the shipped RESP3 server image.
COPY docker/Cargo.toml ./Cargo.toml
COPY docker/Cargo.lock ./Cargo.lock
COPY crates ./crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
        --bin skeg-resp3 -p skeg-server --features tenant-auth \
        ${CARGO_FEATURES:+--features $CARGO_FEATURES} && \
    cp target/release/skeg-resp3 /usr/local/bin/skeg-resp3 && \
    strip /usr/local/bin/skeg-resp3

# ---------- runtime ----------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system --gid 1000 skeg && \
    useradd  --system --uid 1000 --gid 1000 --create-home --home /var/lib/skeg skeg && \
    mkdir -p /var/lib/skeg && \
    chown -R skeg:skeg /var/lib/skeg

COPY --from=builder /usr/local/bin/skeg-resp3  /usr/local/bin/skeg-resp3

USER skeg
WORKDIR /var/lib/skeg
VOLUME ["/var/lib/skeg"]

# The public server image is the one RESP3 binary. It is single-tenant with
# no tenant flags, strict multi-tenant with --tenant-auth --tenant-strict.
EXPOSE 6379

# Docker's default is already SIGTERM; spelling it here makes the image's
# durable shutdown contract part of its metadata rather than an assumption.
STOPSIGNAL SIGTERM

# Listen on all interfaces so Docker port publishing can reach the service.
# Without strict tenant auth the binary refuses this non-loopback bind unless
# the operator explicitly opts in to unauthenticated networking. Production
# invokes this image with --tenant-auth /auth/auth.kdb --tenant-strict.
ENV SKEG_RESP3_ADDR=0.0.0.0:6379 \
    SKEG_DATA_DIR=/var/lib/skeg \
    RUST_LOG=info

ENTRYPOINT ["/usr/local/bin/skeg-resp3"]
