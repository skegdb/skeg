#!/usr/bin/env python3
"""Login load composed with established tenant traffic and disk consolidation.

Always creates its own data and credentials. Host OS is recorded: a Docker
Desktop run is useful local evidence, never relabelled a Linux-host gate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
import time

spec = importlib.util.spec_from_file_location("artifact_smoke", Path(__file__).with_name("smoke-resp3-artifact.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
spec = importlib.util.spec_from_file_location("cgroup_evidence", Path(__file__).with_name("cgroup-evidence.py"))
counters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--memory", type=int, choices=[256, 512], required=True)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    assert args.seconds > 0
    metadata = json.loads(smoke.run("docker", "image", "inspect", args.image))[0]
    proof = dict(status="failed", image_id=metadata["Id"], oci_digests=metadata.get("RepoDigests", []),
                 source_sha=smoke.run("git", "-C", str(smoke.ROOT), "rev-parse", "HEAD"),
                 host_os=os.uname().sysname, limit_mib=args.memory)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="skeg-auth-load-") as scratch:
            root = Path(scratch)
            for directory in ("auth", "data"):
                (root / directory).mkdir()
                (root / directory).chmod(0o777)
            auth = root / "auth/auth.kdb"
            auth.write_bytes(bytes.fromhex((smoke.ROOT / "scripts/fixtures/artifact-auth.hex").read_text()))
            auth.chmod(0o444)
            server = smoke.Server(root, image=metadata["Id"], memory_mib=args.memory)
            stop = threading.Event()
            folding = threading.Event()
            try:
                proof["phase"] = "startup"
                server.start()
                with smoke.Conn(server.port, "alice") as writer, smoke.Conn(server.port, "bob") as reader:
                    proof["phase"] = "populate"
                    writer.cmd("SKEG.VINDEX.CREATE", "load", 128, "disk")
                    vector = struct.pack("<128f", *([0.125] * 128))
                    for i in range(2000):
                        writer.cmd("SKEG.VSET", "load", i, vector)
                        proof["populated"] = i + 1
                    reader.cmd("SET", "sentinel", "bob")

                    def login_load(wrong):
                        counts = dict(accepted=0, backpressure=0, wrongpass=0, attempts_during_fold=0)
                        while not stop.is_set():
                            if folding.is_set():
                                counts["attempts_during_fold"] += 1
                            with smoke.Conn(server.port) as conn:
                                try:
                                    conn.cmd("HELLO", 3, "AUTH", "alice",
                                             "incorrect" if wrong else "alice-smoke-only")
                                    assert not wrong, "incorrect password accepted"
                                    counts["accepted"] += 1
                                except smoke.ReplyError as error:
                                    if str(error).startswith("BACKPRESSURE"):
                                        counts["backpressure"] += 1
                                    elif str(error).startswith("WRONGPASS") and (wrong or "too many" in str(error)):
                                        counts["wrongpass"] += 1
                                    else:
                                        raise
                            # Independent load attempts at a bounded rate, not a
                            # retry policy for a failed application request.
                            stop.wait(0.01)
                        return counts

                    latencies = []
                    proof["phase"] = "concurrent-load"
                    with ThreadPoolExecutor(max_workers=5) as pool:
                        # Establish the maintenance connection before starting
                        # login pressure, so admission of its login isn't the test.
                        maintenance_conn = smoke.Conn(server.port, "alice")
                        try:
                            folding.set()
                            maintenance = pool.submit(maintenance_conn.cmd, "SKEG.VINDEX.CONSOLIDATE", "load")
                            maintenance.add_done_callback(lambda _: folding.clear())
                            jobs = [pool.submit(login_load, i == 3) for i in range(4)]
                            deadline = time.monotonic() + args.seconds
                            try:
                                while time.monotonic() < deadline:
                                    before = time.monotonic()
                                    assert reader.cmd("GET", "sentinel") == b"bob"
                                    writer.cmd("SKEG.VSET", "load", 0, vector)
                                    assert writer.cmd("SKEG.VGET", "load", 0) == vector
                                    latencies.append(time.monotonic() - before)
                                    time.sleep(0.02)
                            finally:
                                stop.set()
                            maintenance.result(timeout=30)
                            proof["login_workers"] = [job.result(timeout=15) for job in jobs]
                        finally:
                            maintenance_conn.close()
                    assert latencies, "established tenants made no progress"
                    assert sum(c["accepted"] for c in proof["login_workers"]) > 0, "no valid login made progress"
                    assert sum(c["attempts_during_fold"] for c in proof["login_workers"]) > 0, "no login attempt overlapped consolidation"
                    assert max(latencies) < 5, "established tenant stalled for 5 seconds"
                    proof.update(operations=len(latencies), max_latency_seconds=max(latencies),
                                 p99_seconds=sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * .99))])
                state = json.loads(smoke.run("docker", "inspect", server.container))[0]
                assert state["HostConfig"]["Memory"] == args.memory * 1024 * 1024
                assert state["HostConfig"]["MemorySwap"] == args.memory * 1024 * 1024
                assert not state["State"]["OOMKilled"]
                kernel = smoke.run("docker", "exec", server.container, "sh", "-c",
                                   'for f in memory.max memory.peak memory.events; do '
                                   'echo "$f:"; cat "/sys/fs/cgroup/$f"; done')
                proof.update(counters.validate(kernel, args.memory))
                server.stop()
                proof["phase"] = "restart"
                server.start()
                with smoke.Conn(server.port, "alice") as conn:
                    assert conn.cmd("SKEG.VGET", "load", 0) == vector
                    assert conn.cmd("SKEG.VGET", "load", 1999) == vector
                with smoke.Conn(server.port, "bob") as conn:
                    assert conn.cmd("GET", "sentinel") == b"bob"
                server.stop()
                proof["status"] = "passed"
                proof["phase"] = "complete"
            except Exception:
                # Capture the failed instance before cleanup; a timeout without
                # server state is not enough evidence to diagnose a stalled run.
                if server.container:
                    try:
                        state = json.loads(smoke.run("docker", "inspect", server.container))[0]
                        proof["container_state"] = state["State"]
                        proof["container_logs"] = smoke.container_logs(server.container)
                        proof["memory_events"] = smoke.run("docker", "exec", server.container,
                                                          "cat", "/sys/fs/cgroup/memory.events")
                        proof["memory_peak_bytes"] = int(smoke.run("docker", "exec", server.container,
                                                                  "cat", "/sys/fs/cgroup/memory.peak"))
                    except Exception as diagnostic_error:
                        proof["diagnostic_error"] = str(diagnostic_error)
                raise
            finally:
                stop.set()
                server.stop(require_success=False)
    except Exception as error:
        proof["error"] = str(error)
        raise
    finally:
        args.evidence.write_text(json.dumps(proof, indent=2) + "\n")
    print(f"PASS auth + tenant traffic + consolidate at {args.memory} MiB: {args.evidence}")


if __name__ == "__main__":
    main()
