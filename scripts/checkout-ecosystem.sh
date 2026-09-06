#!/usr/bin/env bash
# Compatibility entrypoint for candidate workflows; no remote checkout needed.
set -euo pipefail
exec python3 "$(dirname "$0")/check-ecosystem.py"
