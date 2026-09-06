# Deployment contract for 0.8

Skeg is a single-node database. There is no automatic failover, consensus,
replication protocol, leader election or zero-downtime rolling upgrade. Multiple
read-only processes over a completed store are readers, not replicated writers.
Use a local filesystem with reliable advisory locks. NFS/SMB and concurrent raw
file modification are outside the supported storage topology.

## Durability and recovery

An acknowledged operation has the durability level selected for that operation;
process-crash recovery and survival of machine/power failure are different
contracts. Consult [the flush barrier contract](adr-flush-barrier.md). SIGTERM
stops admission, drains connections and waits for shard maintenance and the final
fallible flush. Exit non-zero means that shutdown could not certify its barrier.
`SKEG_SHUTDOWN_TIMEOUT_MS` bounds the connection drain, not disk I/O or the total
time of a large maintenance job. Give the supervisor sufficient termination grace
and alert on forced kills. A forced kill does not prove a successful final flush.

Back up the full data root, auth store and quota sidecar after quiescing writers
and completing the shutdown barrier, or using a separately verified snapshot
procedure. Restore into a new directory and run the same KV/vector queries and
integrity checks before switching traffic. A hot filesystem copy is not a
certified consistent backup. Preserve the old binary and backup for rollback;
do not assume newer on-disk formats support an older executable.

## Tenant exposure and resource limits

Use the canonical `skeg-resp3` binary with `--tenant-auth /auth/auth.kdb
--tenant-strict --admin-tenant admin`. Use unique production credentials, never
the public smoke fixture. Restrict the auth store to its service account.
The native listener is an unauthenticated single-tenant interface and belongs
only on a trusted network. Bind the RESP3 backend to loopback when terminating
TLS on the same host. Do not publish its plaintext port externally.

Password checks use a bounded blocking pool shared by listeners and reserve
their working set against the same memory governor as requests and maintenance.
Saturation returns BACKPRESSURE before hashing; clients should use bounded
backoff with a request deadline, not unbounded retries. Per-IP failed-login
limits remain active. A TCP proxy makes its own source IP visible to the server,
so apply connection limits at the proxy as well and account for shared-IP limits.

`max_vectors` and `max_disk_bytes` are enforced logical quotas. Set both for each
tenant, plus a filesystem/volume cap and free-space alarms: graph/vector/WAL
files, dead records and maintenance scratch are not a complete physical quota.
Use Linux cgroups for the process envelope. A successful macOS or Docker Desktop
smoke does not certify the Linux-host 256/512 MiB maintenance gate.

## TLS proxy example

[`examples/haproxy.cfg`](examples/haproxy.cfg) terminates TLS on port 6380 and
forwards to loopback port 6379. Install a certificate and its private key in the
configured PEM file, restrict its permissions, and arrange certificate renewal.
Run HAProxy in the same network namespace as the backend, or change the upstream
to a private service address protected by firewall rules. Validate the installed
configuration with `haproxy -c -f /etc/haproxy/haproxy.cfg` before reloading.

The example limits concurrent connections; it does not add authentication.
Strict tenant authentication remains mandatory behind TLS. Its TCP backend
health check only detects listener reachability; use the authenticated artifact
smoke and `SKEG.CHECK`/`SKEG.HEALTH` for data and maintenance health.

## Release qualification

Before a full local workspace run, check free space on both the build and
temporary-data volumes (`df -h . "${TMPDIR:-/tmp}"`). Fixtures preallocate
segments; the tiny logical corpus does not imply tiny peak disk use. Leave
generous scratch space (at least 10 GiB beyond existing build output) and
monitor it. Do not prune unrelated containers, volumes or demo data to force
a gate through. An interrupted suite is not a passing qualification.

1. `bash scripts/checkout-ecosystem.sh` verifies vendored adapter provenance.
2. Run fmt, Clippy, workspace tests and `python3 -m unittest discover -s scripts/tests`.
3. `python3 scripts/check-tenant-cli.py --profile release --output evidence/tenant-cli`
   runs 20 serial and 20 parallel suites, stops on the first failure and retains
   phase logs, deadlines, sampled child PIDs and leftover temporary paths.
4. On a Linux host, `bash scripts/check-maintenance-cgroup.sh` runs under real
   256/512 MiB limits and retains exit, OOM and peak-memory evidence.
5. Run `bash scripts/smoke-resp3-artifact.sh --tarball FILE --evidence proof.json`
   on each native platform; its adjacent `.sha256` is mandatory. The OCI form is
   `--image repository@sha256:DIGEST`. It uses the image's actual entrypoint/user.
6. Hosted release jobs download the tarballs, test exact OCI digests, and require
   all five clean-source proofs before promotion. OCI builds request SBOM and
   maximum provenance; `release-evidence.json` binds source and executable hashes
   to stateful validation. A dirty local proof is deliberately rejected.
7. Run `validate_promote=true` on the immutable candidate to prove hosted
   permissions, handoff and verified cleanup. Preparation alone is not this proof.

Use [vendor/README.md](../vendor/README.md) for SDK registry order. The standalone
multi-tenant crate is not published automatically by the engine release.

The optional `model-graveyard/bench/hardening.py` recovery exercise must use the
candidate's absolute `skeg-resp3` path and its own synthetic temporary store,
never `model-graveyard/deploy/index`. Record the binary SHA and raw output;
timed crash injection is supplementary evidence, not a deterministic proof of
every maintenance publication window. Its historical successes do not certify
a newly built candidate.
