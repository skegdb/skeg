#!/usr/bin/env python3
"""Fail on the first bad run; retain phases, process samples and leak evidence."""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def processes():
    result = subprocess.run(["ps", "-axo", "pid=,ppid=,stat=,command="],
                            check=True, text=True, capture_output=True)
    return {int(parts[0]): (int(parts[1]), parts[2], parts[3])
            for line in result.stdout.splitlines()
            if len(parts := line.strip().split(None, 3)) == 4}


def descendants(table, root):
    found = {root}
    while True:
        more = {pid for pid, (parent, _, _) in table.items() if parent in found}
        if more <= found:
            return found
        found |= more


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--deadline", type=float, default=60)
    parser.add_argument("--profile", choices=["debug", "release"], default="debug")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.deadline <= 0:
        parser.error("positive runs and deadline required")
    args.output.mkdir(parents=True, exist_ok=True)
    command = ["cargo", "test", "-p", "skeg-server", "--locked", "--test",
               "tenant_cli", "--no-run", "--message-format=json"]
    if args.profile == "release":
        command.append("--release")
    built = subprocess.run(command, check=True, capture_output=True, text=True)
    binary = None
    for line in built.stdout.splitlines():
        event = json.loads(line)
        if event.get("executable") and event.get("target", {}).get("name") == "tenant_cli":
            binary = event["executable"]
    if not binary:
        raise RuntimeError("Cargo did not name tenant_cli executable")
    results = []
    for mode, threads in [("serial", 1), ("parallel", 4)]:
        for index in range(args.runs):
            name = f"{mode}-{index + 1:02}"
            log = args.output / f"{name}.log"
            with tempfile.TemporaryDirectory(prefix="skeg-tenant-cli-") as scratch:
                env = dict(os.environ, TMPDIR=scratch, TMP=scratch, TEMP=scratch)
                env["RUST_BACKTRACE"] = "1"
                observed = set()
                samples = []
                started = time.monotonic()
                with log.open("w") as stream:
                    child = subprocess.Popen([binary, "--nocapture", f"--test-threads={threads}"],
                                             env=env, stdout=stream, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    sampled = False
                    timed_out = False
                    while child.poll() is None:
                        table = processes()
                        observed |= descendants(table, child.pid)
                        elapsed = time.monotonic() - started
                        if elapsed > 5 and not sampled:
                            sampled = True
                            samples.append({pid: table[pid] for pid in observed if pid in table})
                            if os.uname().sysname == "Darwin":
                                subprocess.run(["sample", str(child.pid), "2", "1", "-file",
                                                str(args.output / f"{name}.sample.txt")],
                                               timeout=10, stdout=stream, stderr=stream)
                        if elapsed > args.deadline:
                            timed_out = True
                            os.killpg(child.pid, signal.SIGKILL)
                            break
                        time.sleep(0.05)
                    child.wait(timeout=10)
                table = processes()
                # The child may have escaped between two process-table samples;
                # the fixture also records its PID immediately after spawn.
                logged = log.read_text()
                observed |= {int(pid) for pid in re.findall(r"phase: (?:spawned|ready|reaped) pid=(\d+)", logged)}
                leaks = {pid: table[pid] for pid in observed if pid != child.pid and pid in table}
                leftovers = [str(p.relative_to(scratch)) for p in Path(scratch).rglob("*")]
                record = dict(mode=mode, run=index + 1, seconds=time.monotonic() - started,
                              exit=child.returncode, timeout=timed_out, leaked_processes=leaks,
                              leftover_paths=leftovers, samples=samples)
                results.append(record)
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{name}: {record['seconds']:.2f}s exit={child.returncode} "
                      f"leaks={len(leaks)} paths={len(leftovers)}", flush=True)
                if child.returncode or timed_out or leaks or leftovers:
                    # Only processes started by this run; never kill other test/server sessions.
                    for pid in leaks:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    raise SystemExit(f"FAILED {name}; evidence: {args.output}")


if __name__ == "__main__":
    main()
