#!/usr/bin/env python3
"""Validate the conformance cases against a live skeg-resp3 server.

This is the maintenance tool for the case files, not a client runner: it
speaks the wire directly so a case can never be "proved" by the same client
bug it is meant to catch. Each client repo ships its own runner that drives
its own SDK over these same cases.

    python3 validate.py --bin /path/to/skeg-resp3 [--profile anon]

Exits 0 when every case of the profile matches, 1 otherwise.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def encode_arg(a: str) -> bytes:
    """Case args are text by default, with two escapes: `b64:` for arbitrary
    bytes and `f32:[..]` for a little-endian f32 vector."""
    if a.startswith("b64:"):
        return base64.b64decode(a[4:])
    if a.startswith("f32:"):
        vals = json.loads(a[4:])
        return struct.pack(f"<{len(vals)}f", *[float(v) for v in vals])
    return a.encode()


def encode_command(args: list[str]) -> bytes:
    out = [b"*%d\r\n" % len(args)]
    for a in args:
        b = encode_arg(a)
        out.append(b"$%d\r\n%s\r\n" % (len(b), b))
    return b"".join(out)


class Reply:
    """A parsed RESP reply. `kind` is the wire marker, `value` the payload."""

    def __init__(self, kind: str, value):
        self.kind = kind
        self.value = value

    def __repr__(self) -> str:
        return f"{self.kind}:{self.value!r}"


class Conn:
    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.buf = b""

    def close(self) -> None:
        self.sock.close()

    def _fill(self) -> None:
        chunk = self.sock.recv(65536)
        if not chunk:
            raise EOFError("server closed the connection")
        self.buf += chunk

    def _line(self) -> bytes:
        while b"\r\n" not in self.buf:
            self._fill()
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line

    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n + 2:
            self._fill()
        data, self.buf = self.buf[:n], self.buf[n + 2 :]
        return data

    def read(self) -> Reply:
        line = self._line()
        marker, body = chr(line[0]), line[1:]
        if marker == "+":
            return Reply("simple", body.decode())
        if marker == "-":
            return Reply("error", body.decode())
        if marker == ":":
            return Reply("integer", int(body))
        if marker == ",":
            return Reply("double", float(body))
        if marker == "#":
            return Reply("boolean", body == b"t")
        if marker == "_":
            return Reply("null", None)
        if marker == "!":
            return Reply("error", self._exact(int(body)).decode())
        if marker in "$=":
            n = int(body)
            if n < 0:
                return Reply("null", None)
            return Reply("bulk", self._exact(n))
        if marker in "*~>":
            n = int(body)
            if n < 0:
                return Reply("null", None)
            return Reply("array", [self.read() for _ in range(n)])
        if marker == "%":
            n = int(body)
            return Reply("map", [(self.read(), self.read()) for _ in range(n)])
        if marker == "(":
            return Reply("bignum", body.decode())
        raise ValueError(f"unknown RESP marker {marker!r} in {line!r}")

    def call(self, args: list[str]) -> Reply:
        self.sock.sendall(encode_command(args))
        return self.read()


def retryability(reply: Reply) -> bool:
    """Does this reply tell the client the same command is worth resending?

    The first word of a RESP error IS its code, and `BACKPRESSURE` is the only
    one that means "no room right now". Everything else - including a plain
    `ERR` in front of a condition that clears - means give up.
    """
    return reply.kind == "error" and reply.value.split(" ", 1)[0] == "BACKPRESSURE"


def check(reply: Reply, want: dict) -> str | None:
    """Return None when the reply satisfies `want`, else a failure reason."""
    # Checked first and alongside the rest: retryability is a property of the
    # SAME reply another matcher is about, not a matcher of its own. The
    # native cases carry the identical key against the error code byte, which
    # is what makes the two wires comparable case by case.
    if "retryable" in want and retryability(reply) != want["retryable"]:
        return (
            f"reply {reply} is {'retryable' if retryability(reply) else 'permanent'}, "
            f"want {'retryable' if want['retryable'] else 'permanent'}"
        )
    if "any" in want:
        return None if reply.kind != "error" else f"unexpected error {reply.value!r}"
    if "error_contains" in want:
        if reply.kind != "error":
            return f"expected an error, got {reply}"
        if want["error_contains"] not in reply.value:
            return f"error {reply.value!r} lacks {want['error_contains']!r}"
        return None
    if reply.kind == "error":
        return f"unexpected error {reply.value!r}"
    if "simple" in want:
        got = reply.value.decode() if reply.kind == "bulk" else reply.value
        return None if got == want["simple"] else f"want simple {want['simple']!r}, got {reply}"
    if "bulk" in want:
        if reply.kind != "bulk":
            return f"want bulk, got {reply}"
        return None if reply.value == encode_arg(want["bulk"]) else f"bulk {reply.value!r} != {want['bulk']!r}"
    if "bulk_contains" in want:
        if reply.kind != "bulk":
            return f"want bulk, got {reply}"
        text = reply.value.decode(errors="replace")
        missing = [f for f in want["bulk_contains"] if f not in text]
        return None if not missing else f"bulk lacks {missing!r}"
    if "integer" in want:
        if reply.kind != "integer":
            return f"want integer, got {reply}"
        return None if reply.value == want["integer"] else f"want {want['integer']}, got {reply.value}"
    if "integer_min" in want:
        if reply.kind != "integer":
            return f"want integer, got {reply}"
        return None if reply.value >= want["integer_min"] else f"{reply.value} < {want['integer_min']}"
    if "double" in want:
        return None if reply.kind == "double" else f"want double, got {reply}"
    if "null" in want:
        return None if reply.kind == "null" else f"want null, got {reply}"
    if "array_len" in want or "items" in want:
        if reply.kind != "array":
            return f"want array, got {reply}"
        if "array_len" in want and len(reply.value) != want["array_len"]:
            return f"want {want['array_len']} items, got {len(reply.value)}"
        for i, item_want in enumerate(want.get("items", [])):
            reason = check(reply.value[i], item_want)
            if reason:
                return f"item[{i}]: {reason}"
        return None
    if "map_has" in want or "map_field" in want:
        if reply.kind != "map":
            return f"want map, got {reply}"
        fields = {
            (k.value.decode() if isinstance(k.value, bytes) else str(k.value)): v
            for k, v in reply.value
        }
        missing = [k for k in want.get("map_has", []) if k not in fields]
        if missing:
            return f"map lacks {missing!r}"
        for k, v in want.get("map_field", {}).items():
            if k not in fields:
                return f"map lacks {k!r}"
            got = fields[k].value
            got = int(got) if isinstance(got, (int, float)) else got
            if got != v:
                return f"map[{k}] = {got!r}, want {v!r}"
        return None
    return f"case has no known matcher: {want!r}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_tcp(port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to the skeg-resp3 binary")
    ap.add_argument("--cases", default=str(HERE / "resp3-cases.jsonl"))
    ap.add_argument("--profile", default="anon", help="run cases of this profile (plus unlabelled ones)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    cases = [c for c in cases if c.get("profile", args.profile) == args.profile]

    port = free_port()
    data_dir = Path(tempfile.mkdtemp(prefix="skeg-conformance-"))
    env = {**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "warn")}
    proc = subprocess.Popen(
        [args.bin, "--mode", "rw", "--addr", f"127.0.0.1:{port}", "--data-dir", str(data_dir)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_tcp(port):
            print("server did not start", file=sys.stderr)
            return 1
        conn = Conn(port)
        conn.call(["HELLO", "3"])  # cases assume RESP3 typing unless fresh_conn

        failures, known_bugs = [], []
        for case in cases:
            # HELLO cases renegotiate, so they get their own connection rather
            # than leaving the shared one on an unexpected protocol version.
            fresh = case["cmd"][0].upper() == "HELLO"
            c = Conn(port) if fresh else conn
            try:
                reply = c.call(case["cmd"])
                reason = check(reply, case["want"])
            except Exception as e:  # noqa: BLE001 - a crash is a case failure
                reply, reason = None, f"{type(e).__name__}: {e}"
            finally:
                if fresh:
                    c.close()
            # A case carrying `bug` documents a server defect: it is expected
            # to fail. When it starts passing the defect is fixed and the
            # marker must go, so that is reported as loudly as a failure.
            if "bug" in case:
                if reason:
                    known_bugs.append(case["id"])
                    print(f"bug  {case['id']}: {case['bug']}")
                else:
                    failures.append((case["id"], "marked as a server bug but now passes; drop the marker"))
                    print(f"FIXED {case['id']}: drop the `bug` marker")
                continue
            if reason:
                failures.append((case["id"], reason))
                print(f"FAIL {case['id']}: {reason}")
            elif args.verbose:
                print(f"ok   {case['id']} -> {reply}")

        passed = len(cases) - len(failures) - len(known_bugs)
        print(f"\n{passed}/{len(cases)} cases passed (profile={args.profile}), "
              f"{len(known_bugs)} known server bug(s)")
        return 1 if failures else 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
