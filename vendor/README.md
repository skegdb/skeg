# Candidate adapter sources

These four Apache-2.0 crates are exact source snapshots from skegdb's candidate
adapter repositories. Origin commits and SHA-256 of every copied file are in
`SOURCES.json`. Only the two workspace manifests are narrowed to the included
members; crate manifests, implementation, tests and benchmarks are unchanged.

The registry versions of these candidates are not published yet. Vendoring
makes a fresh engine checkout buildable without sibling directories or private
remote commits. `scripts/check-ecosystem.py` rejects drift in CI.

Publish order once engine crates are available on the registry:

1. skeg-vector 0.2 and skeg-resp3 0.3 (engine release order).
2. skeg-rigging 0.1.5, then skeg-rigging-skeg 0.1.5.
3. skeg-rigging-net 0.1.2, then skeg-rigging-net-resp3 0.1.2.
4. skeg-multi-tenant 0.1.1, after an extracted package test with `live-attach`.

Do not publish a package solely because workspace tests with these patches pass.
The registry consumer graph must contain one skeg-vector generation. Once that
graph is published and independently verified, remove the adapter patches and
these snapshots in the same change, regenerate Cargo.lock and repeat package CI.
