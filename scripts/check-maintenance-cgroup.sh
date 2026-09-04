#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: the maintenance memory gate requires a Linux cgroup" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "error: Docker is required to create the 256/512 MiB cgroups" >&2
  exit 2
fi

repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"

build_output=$(cargo test -p skeg-server --test maintenance_cgroup \
  --release --locked --no-run 2>&1)
printf '%s\n' "$build_output"
test_bin=$(sed -n \
  's|^[[:space:]]*Executable tests/maintenance_cgroup.rs (\(.*\))$|\1|p' \
  <<<"$build_output" | tail -n 1)
if [[ -z "$test_bin" ]]; then
  echo "error: cargo did not produce the maintenance_cgroup test binary" >&2
  exit 1
fi
test_bin="/work/${test_bin#./}"

for limit in 256 512; do
  echo "maintenance cgroup gate: ${limit} MiB"
  docker run --rm \
    --memory="${limit}m" \
    --memory-swap="${limit}m" \
    --pids-limit=512 \
    --network=none \
    --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=64m \
    --mount "type=bind,src=${repo},dst=/work,readonly" \
    --workdir /work \
    --env "SKEG_CGROUP_LIMIT_MIB=${limit}" \
    ubuntu:24.04 \
    "$test_bin" \
    --ignored \
    --exact ingest_and_fold_stay_alive_under_the_cgroup_budget \
    --nocapture
done
