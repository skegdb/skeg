#!/usr/bin/env python3
"""Every crate whose source changed since the previous release tag must carry a
version that is not yet on crates.io. Run before anything is published: a
changed crate with an already-published version would make `cargo publish`
answer "already exists", which the publish job treats as success for a crate
it shipped earlier in the same run - and a tag would go green without shipping
the code it tested.

Usage: scripts/check-release-versions.py [--prev-tag TAG] [--tag vX.Y.Z]
With --tag (the release workflow passes the pushed tag) the tag must equal
"v" + skeg-server's version, and the CHANGELOG's "Upgrading" table must name
the same versions the manifests carry.
Exit 1 with one line per offending crate. Reads the sparse index at
index.crates.io (no auth); a crate that is not on the index at all is fine.
"""
import json, os, re, subprocess, sys, urllib.request

CRATES = ["skeg-proto", "skeg-simd", "skeg-platform", "skeg-telemetry", "skeg-resp3",
          "skeg-core", "skeg-vector", "skeg-tenant", "skeg-server", "skeg-server-tenant"]
# skeg-multi-tenant is not published by this release (skeg-rigging-skeg 0.1.4 pins the
# engine to ^0.1); it rejoins the list when the workflow's publish loop does.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True)

def prev_tag(explicit):
    if explicit:
        return explicit
    head = sh("git", "describe", "--tags", "--exact-match", "HEAD").stdout.strip()
    # Reachable tags only: a newer tag on a branch this commit never merged
    # would make the diff lie about what changed (same rule as release.yml).
    tags = [t for t in sh("git", "tag", "--merged", "HEAD", "--sort=-creatordate").stdout.split() if t.startswith("v") and t != head]
    return tags[0] if tags else ""

def local_version(crate):
    meta = json.loads(sh("cargo", "metadata", "--no-deps", "--format-version", "1").stdout)
    return next(p["version"] for p in meta["packages"] if p["name"] == crate)

def index_path(name):
    n = len(name)
    if n == 1: return f"1/{name}"
    if n == 2: return f"2/{name}"
    if n == 3: return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"

def published_versions(name):
    req = urllib.request.Request(f"https://index.crates.io/{index_path(name)}",
                                 headers={"User-Agent": "skeg-release-gate"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return {json.loads(l)["vers"] for l in r.read().decode().splitlines() if l.strip()}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return set()
        raise

def changelog_table_mismatches(versions):
    """Rows of the CHANGELOG 'Upgrading' table whose 'to' column disagrees with the manifest."""
    bad = []
    try:
        text = open(os.path.join(ROOT, "CHANGELOG.md")).read()
    except FileNotFoundError:
        return ["CHANGELOG.md missing"]
    for m in re.finditer(r"^\| (skeg-[a-z-]+) \| [^|]* \| ([^|]*) \|", text, re.M):
        crate, to = m.group(1), m.group(2).strip().strip("*")
        if "not published" in to or crate not in versions:
            continue
        if to != versions[crate]:
            bad.append(f"CHANGELOG table says {crate} -> {to}, manifest says {versions[crate]}")
    return bad

def main():
    args = sys.argv[1:]
    tag = prev_tag(args[args.index("--prev-tag") + 1] if "--prev-tag" in args else "")
    pushed = args[args.index("--tag") + 1] if "--tag" in args else ""
    print(f"previous tag: {tag or '(none: every crate counts as changed)'}")
    bad = []
    changed = []
    for c in CRATES:
        if tag and sh("git", "diff", "--quiet", tag, "HEAD", "--", f"crates/{c}").returncode == 0:
            continue
        changed.append(c)
        v = local_version(c)
        if v in published_versions(c):
            bad.append(f"{c}: crates/{c} changed since {tag or 'the beginning'} but {v} is already on crates.io")
    print("changed: " + (", ".join(changed) or "(none)"))
    versions = {c: local_version(c) for c in CRATES}
    if pushed:
        want = "v" + versions["skeg-server"]
        if pushed != want:
            bad.append(f"tag {pushed} does not name the product version: expected {want} (skeg-server {versions['skeg-server']})")
        bad += changelog_table_mismatches(versions)
    for b in bad:
        print("FAIL: " + b)
    if bad:
        return 1
    print("ok: every changed crate carries an unpublished version")
    return 0

if __name__ == "__main__":
    sys.exit(main())
