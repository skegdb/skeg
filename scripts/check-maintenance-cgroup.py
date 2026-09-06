#!/usr/bin/env python3
"""Linux-host maintenance qualification with retained OOM, limit and peak evidence."""
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import uuid

root = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cgroup_evidence", root / "scripts/cgroup-evidence.py")
counters = importlib.util.module_from_spec(spec)
spec.loader.exec_module(counters)
if os.uname().sysname != "Linux":
    raise SystemExit("requires a Linux host; Docker Desktop is development evidence only")
output = Path(os.environ.get("SKEG_CGROUP_EVIDENCE", "evidence/cgroup")).resolve()
output.mkdir(parents=True, exist_ok=True)
built = subprocess.run(["cargo", "test", "-p", "skeg-server", "--test", "maintenance_cgroup",
                        "--release", "--locked", "--no-run", "--message-format=json"],
                       cwd=root, check=True, capture_output=True, text=True)
binary = None
for line in built.stdout.splitlines():
    event = json.loads(line)
    if event.get("executable") and event.get("target", {}).get("name") == "maintenance_cgroup":
        binary = str(Path(event["executable"]).resolve())
assert binary, "Cargo did not produce the gate executable"

for limit in (256, 512):
    name = f"skeg-maintenance-{uuid.uuid4().hex}"
    with open(binary, "rb") as executable:
        digest = hashlib.sha256()
        for chunk in iter(lambda: executable.read(1024 * 1024), b""):
            digest.update(chunk)
        executable_sha = digest.hexdigest()
    evidence = {"limit_mib": limit, "status": "failed", "host_os": os.uname().sysname,
                "executable_sha256": executable_sha,
                "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    with tempfile.TemporaryDirectory(prefix="skeg-cgroup-data-") as data:
        command = ["docker", "run", "--name", name, f"--memory={limit}m",
                   f"--memory-swap={limit}m", "--pids-limit=512", "--network=none",
                   "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m",
                   "--mount", f"type=bind,src={binary},dst=/gate,readonly",
                   "--mount", f"type=bind,src={data},dst=/data",
                   "--env", "TMPDIR=/data", "--env", f"SKEG_CGROUP_LIMIT_MIB={limit}",
                   "ubuntu:24.04", "sh", "-c",
                   '/gate --ignored --exact ingest_and_fold_stay_alive_under_the_cgroup_budget --nocapture; '
                   'code=$?; for f in memory.max memory.peak memory.events; do '
                   'echo "$f:"; cat "/sys/fs/cgroup/$f"; done; exit "$code"']
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            (output / f"{limit}.log").write_text(result.stdout + result.stderr)
            state = json.loads(subprocess.run(["docker", "inspect", name], check=True,
                                             capture_output=True, text=True).stdout)[0]
            evidence.update(exit=result.returncode, container_state=state["State"],
                            memory_bytes=state["HostConfig"]["Memory"],
                            memory_swap_bytes=state["HostConfig"]["MemorySwap"])
            assert state["HostConfig"]["Memory"] == limit * 1024 * 1024
            assert state["HostConfig"]["MemorySwap"] == limit * 1024 * 1024
            assert not state["State"]["OOMKilled"], "cgroup OOM killed the gate"
            assert result.returncode == 0, f"gate failed; see {output / f'{limit}.log'}"
            assert "1 passed" in result.stdout, "gate ran zero tests"
            evidence.update(counters.validate(result.stdout, limit))
            evidence["status"] = "passed"
            print(f"PASS maintenance {limit} MiB", flush=True)
        except subprocess.TimeoutExpired as error:
            evidence["error"] = "maintenance deadline exceeded"
            (output / f"{limit}.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
            raise
        except Exception as error:
            evidence["error"] = str(error)
            raise
        finally:
            (output / f"{limit}.json").write_text(json.dumps(evidence, indent=2) + "\n")
            subprocess.run(["docker", "rm", "--force", name], check=True, capture_output=True)
