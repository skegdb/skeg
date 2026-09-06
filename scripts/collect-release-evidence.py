#!/usr/bin/env python3
"""Refuse promotion without all five native artifact proofs on this exact SHA."""
import argparse
import json
from pathlib import Path
import re


def collect(directory, source_sha):
    names = [f"tarball-{target}.json" for target in (
        "aarch64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu")]
    names += ["image-0.json", "image-1.json"]
    proofs = []
    for name in names:
        proof = json.loads((directory / name).read_text())
        assert proof["status"] == "passed", f"failed smoke: {name}"
        assert proof["source_sha"] == source_sha, f"wrong source: {name}"
        assert proof["source_dirty"] is False, f"dirty source: {name}"
        assert re.fullmatch(r"[0-9a-f]{64}", proof["executable_sha256"]), name
        if name.startswith("image"):
            assert proof["oci_digests"], f"missing published digest: {name}"
        else:
            assert re.fullmatch(r"[0-9a-f]{64}", proof["tarball_sha256"]), name
            assert re.fullmatch(r"[0-9a-f]{64}", proof["native_executable_sha256"]), name
        assert proof["inventory"], f"missing persisted state proof: {name}"
        proofs.append(proof)
    return dict(source_sha=source_sha, artifacts=proofs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("source_sha")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9a-f]{40}", args.source_sha)
    args.output.write_text(json.dumps(collect(args.directory, args.source_sha), indent=2) + "\n")
