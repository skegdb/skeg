#!/usr/bin/env bash
# Common stateful contract for extracted tarballs and exact OCI images.
set -euo pipefail
exec python3 "$(dirname "$0")/smoke-resp3-artifact.py" "$@"
