#!/usr/bin/env python3
"""Verify the vendored candidate graph without network or sibling checkouts."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / "vendor/SOURCES.json").read_text())
for path, expected in manifest["files"].items():
    actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"vendored source changed: {path}; update provenance deliberately")
actual_paths = {str(p.relative_to(root)) for repo in manifest["repositories"]
                for p in (root / "vendor" / repo).rglob("*") if p.is_file()}
assert actual_paths == set(manifest["files"]), "unexpected or missing vendored source files"
for name, commit in manifest["repositories"].items():
    print(f"verified {name} {commit}")
