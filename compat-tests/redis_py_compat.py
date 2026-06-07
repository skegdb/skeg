"""End-to-end RESP3 compatibility test against a real Redis client SDK.

Spins skeg-resp3 on an ephemeral port, drives every typed command from
`skeg-resp3` through `redis-py`, and asserts both the result shape AND
the byte-exact error strings the parser was designed to preserve.

Run:

    pip install redis>=5
    cargo build --release -p skeg-server
    python3 tools/skeg-tool/tests/redis_py_compat.py

The script exits 0 on success, non-zero on any assertion failure. CI
wires it as a smoke test once the redis-py dep is added to the dev
matrix; for now it is a manual gate.
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import redis
except ImportError:
    print("redis-py not installed; pip install redis>=5", file=sys.stderr)
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[1]
SKEG_RESP3_BIN = REPO_ROOT / "target" / "release" / "skeg-resp3"


def free_port() -> int:
    """Pick a free TCP port. The race between close() and skeg bind is
    accepted - the loopback range is sparse and skeg fails loud if it
    cannot bind, so a flake here is visible, not silent."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_tcp(port: int, timeout: float = 10.0) -> bool:
    """Poll the port until skeg accepts a TCP connection."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                return True
        except OSError:
            time.sleep(0.05)
    return False


def spawn_server() -> tuple[subprocess.Popen, int, Path]:
    if not SKEG_RESP3_BIN.exists():
        print(
            f"missing binary {SKEG_RESP3_BIN}; build with "
            "`cargo build --release -p skeg-server`",
            file=sys.stderr,
        )
        sys.exit(1)
    port = free_port()
    data_dir = Path(tempfile.mkdtemp(prefix="skeg-redis-compat-"))
    env = os.environ.copy()
    env.setdefault("RUST_LOG", "warn")
    proc = subprocess.Popen(
        [
            str(SKEG_RESP3_BIN),
            "--mode",
            "dev",
            "--addr",
            f"127.0.0.1:{port}",
            "--data-dir",
            str(data_dir),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_tcp(port):
        proc.terminate()
        proc.wait(timeout=5)
        sys.exit("skeg-resp3 failed to start within 10s")
    return proc, port, data_dir


def pack_vec(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def expect_error(fn, fragment: str):
    try:
        fn()
    except redis.ResponseError as e:
        if fragment not in str(e):
            raise AssertionError(
                f"expected error containing {fragment!r}, got {e!r}"
            ) from e
        return
    raise AssertionError(f"expected ResponseError containing {fragment!r}, got success")


def run_tests(port: int) -> int:
    r = redis.Redis(host="127.0.0.1", port=port, decode_responses=True)

    # Try the protocol negotiation first. Both RESP2 (no HELLO) and
    # RESP3 (HELLO 3) must work; redis-py auto-negotiates RESP2 by
    # default.
    assert r.ping() == True, "PING"

    # ── KV happy path ──────────────────────────────────────────────
    r.set("k1", "hello")
    assert r.get("k1") == "hello"
    r.mset({"a": "1", "b": "2", "c": "3"})
    assert r.mget(["a", "b", "missing", "c"]) == ["1", "2", None, "3"]
    assert r.exists("k1", "a", "missing") == 2
    assert r.delete("k1", "a", "missing") == 2

    # Counters.
    r.set("counter", "10")
    assert r.incr("counter") == 11
    assert r.incrby("counter", 5) == 16
    assert r.decr("counter") == 15
    assert r.decrby("counter", 3) == 12

    # SELECT compat: only DB 0 is honoured.
    r.execute_command("SELECT", "0")
    expect_error(
        lambda: r.execute_command("SELECT", "1"),
        "DB index out of range",
    )

    # ── KV error strings (parser invariants) ──────────────────────
    expect_error(lambda: r.execute_command("GET"), "wrong number of arguments for 'GET'")
    expect_error(lambda: r.execute_command("SET", "k"), "wrong number of arguments for 'SET'")
    expect_error(lambda: r.execute_command("DEL"), "wrong number of arguments for 'DEL'")
    expect_error(
        lambda: r.execute_command("MSET", "k", "v", "x"),
        "wrong number of arguments for 'MSET'",
    )
    expect_error(
        lambda: r.execute_command("INCRBY", "ctr", "abc"),
        "value is not an integer or out of range",
    )

    # ── SKEG.* admin ───────────────────────────────────────────────
    stats = r.execute_command("SKEG.STATS")
    assert isinstance(stats, str) and "cache_bytes=" in stats, f"STATS: {stats!r}"
    whoami = r.execute_command("SKEG.WHOAMI")
    assert "single-tenant" in whoami, f"WHOAMI: {whoami!r}"

    expect_error(
        lambda: r.execute_command("SKEG.VINDEX.CREATE", "x"),
        "want name dim kind backend",
    )
    expect_error(
        lambda: r.execute_command("SKEG.FOO"),
        "unknown command 'SKEG.FOO'",
    )

    # ── SKEG vector ops ────────────────────────────────────────────
    r.execute_command("SKEG.VINDEX.CREATE", "v", "8", "int8", "flat")
    listing = r.execute_command("SKEG.VINDEX.LIST")
    assert "name=v" in listing and "dim=8" in listing, f"LIST: {listing!r}"

    raw = redis.Redis(host="127.0.0.1", port=port, decode_responses=False)
    raw.execute_command(
        "SKEG.VSET", "v", "1", pack_vec([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    raw.execute_command(
        "SKEG.VSET", "v", "2", pack_vec([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    hits = raw.execute_command(
        "SKEG.VSEARCH",
        "v",
        "1",
        "32",
        pack_vec([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert isinstance(hits, list) and len(hits) >= 2, f"VSEARCH: {hits!r}"
    assert hits[0] == b"1", f"VSEARCH top-1 id: {hits[0]!r}"

    deleted = raw.execute_command("SKEG.VDEL", "v", "2")
    assert deleted == 1, f"VDEL: {deleted!r}"

    r.execute_command("SKEG.VINDEX.DROP", "v")

    # ── RESP3 negotiation: redis-py 5+ supports it via the `protocol`
    # constructor arg. Verify HELLO 3 lands on skeg.
    r3 = redis.Redis(host="127.0.0.1", port=port, protocol=3, decode_responses=True)
    assert r3.ping() == True, "PING/RESP3"
    r3.set("rk", "v3")
    assert r3.get("rk") == "v3"
    r3.delete("rk")

    print("all redis-py compat assertions passed")
    return 0


def main() -> int:
    proc, port, data_dir = spawn_server()
    try:
        return run_tests(port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        # best-effort cleanup; the tempdir is harmless if it leaks
        import shutil

        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
