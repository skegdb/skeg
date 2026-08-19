#!/usr/bin/env python3
"""Validate the native-protocol conformance cases against a live skeg server.

Same role as validate.py, for the binary protocol: it builds frames byte by
byte so a case cannot be "proved" by the client bug it exists to catch.

    python3 validate_native.py --bin /path/to/skeg

Header (24 bytes, little-endian):
    [magic u16][version u8][op u8][flags u32][req_id u64][payload_len u32][reserved u32]
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
MAGIC = 0x564B
HEADER_LEN = 24

OPS = {
    "get": 0x01, "set": 0x02, "del": 0x03, "mget": 0x04, "mset": 0x05,
    "exists": 0x06, "mexists": 0x07,
    "vindex_create": 0x10, "vindex_drop": 0x11, "vset": 0x12, "vget": 0x13,
    "vdel": 0x14, "vsearch": 0x15, "vindex_list": 0x16,
    "ping": 0x80, "stats": 0x81, "flush": 0x82, "shards": 0x83,
    "native_hello": 0x84,
}
OP_OK, OP_ERR = 0xC0, 0xC1


def as_bytes(v: str) -> bytes:
    return base64.b64decode(v[4:]) if v.startswith("b64:") else v.encode()


def name_field(name: str) -> bytes:
    b = name.encode()
    return struct.pack("<H", len(b)) + b


def vector_field(values: list) -> bytes:
    return struct.pack("<I", len(values)) + struct.pack(f"<{len(values)}f", *[float(v) for v in values])


def build_payload(op: str, a: dict) -> bytes:
    """Payload layouts come from skeg-proto's request.rs and vector.rs."""
    if op in ("ping", "stats", "shards", "flush", "native_hello", "vindex_list"):
        return b""
    if op in ("get", "del", "exists"):
        k = as_bytes(a["key"])
        return struct.pack("<H", len(k)) + k
    if op in ("set", "mset"):
        k, v = as_bytes(a["key"]), as_bytes(a["value"])
        return struct.pack("<HI", len(k), len(v)) + k + v
    if op in ("mget", "mexists"):
        keys = [as_bytes(k) for k in a["keys"]]
        return struct.pack("<I", len(keys)) + b"".join(struct.pack("<H", len(k)) + k for k in keys)
    if op == "vindex_create":
        return name_field(a["name"]) + struct.pack("<IBB", a["dim"], a["kind"], a["backend"])
    if op == "vindex_drop":
        return name_field(a["name"])
    if op in ("vget", "vdel"):
        return name_field(a["name"]) + struct.pack("<Q", a["id"])
    if op == "vset":
        return name_field(a["name"]) + struct.pack("<Q", a["id"]) + vector_field(a["vector"])
    if op == "vsearch":
        return name_field(a["name"]) + struct.pack("<I", a["k"]) + vector_field(a["query"])
    raise ValueError(f"no payload builder for op {op!r}")


def build_frame(op: str, version: int, req_id: int, payload: bytes) -> bytes:
    return struct.pack("<HBBIQII", MAGIC, version, OPS[op], 0, req_id, len(payload), 0) + payload


class Reply:
    def __init__(self, version: int, op: int, req_id: int, payload: bytes):
        self.version, self.op, self.req_id, self.payload = version, op, req_id, payload

    @property
    def is_error(self) -> bool:
        return self.op == OP_ERR

    def error_text(self) -> str:
        # Err payload: [u8 code][u8 msg_len][msg]
        return self.payload[2:].decode(errors="replace") if len(self.payload) >= 2 else ""

    def error_code(self) -> int:
        return self.payload[0] if self.payload else 0

    def __repr__(self) -> str:
        tag = "ERR" if self.is_error else ("OK" if self.op == OP_OK else hex(self.op))
        body = self.error_text() if self.is_error else self.payload[:32]
        return f"v{self.version}/{tag}/{body!r}"


class Conn:
    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.buf = b""

    def close(self) -> None:
        self.sock.close()

    def _need(self, n: int) -> None:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("server closed the connection")
            self.buf += chunk

    def call(self, frame: bytes) -> Reply:
        self.sock.sendall(frame)
        self._need(HEADER_LEN)
        magic, version, op, _flags, req_id, plen, _res = struct.unpack("<HBBIQII", self.buf[:HEADER_LEN])
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic:#x}")
        self._need(HEADER_LEN + plen)
        payload = self.buf[HEADER_LEN : HEADER_LEN + plen]
        self.buf = self.buf[HEADER_LEN + plen :]
        return Reply(version, op, req_id, payload)


def decode_value(payload: bytes) -> bytes:
    n = struct.unpack("<I", payload[:4])[0]
    return payload[4 : 4 + n]


def decode_mget(payload: bytes):
    n = struct.unpack("<I", payload[:4])[0]
    out, pos = [], 4
    for _ in range(n):
        status = payload[pos]
        pos += 1
        if status == 0:
            vlen = struct.unpack("<I", payload[pos : pos + 4])[0]
            pos += 4
            out.append(payload[pos : pos + vlen])
            pos += vlen
        else:
            out.append(None)
    return out


def decode_vindex_list(payload: bytes) -> list[str]:
    n = struct.unpack("<I", payload[:4])[0]
    names, pos = [], 4
    for _ in range(n):
        nlen = struct.unpack("<H", payload[pos : pos + 2])[0]
        pos += 2
        names.append(payload[pos : pos + nlen].decode(errors="replace"))
        pos += nlen + 4 + 1 + 1 + 8  # dim, kind, backend, n_vectors
    return names


def decode_hits(payload: bytes):
    n = struct.unpack("<I", payload[:4])[0]
    return [struct.unpack_from("<Qf", payload, 4 + i * 12) for i in range(n)]


def check(reply: Reply, want: dict, case: dict) -> str | None:
    # Checked first, and for errors too: the server promises every reply keeps
    # the request's frame version.
    if "version" in want and reply.version != want["version"]:
        return f"reply version {reply.version} != {want['version']}"
    if "req_id" in want and reply.req_id != want["req_id"]:
        return f"req_id {reply.req_id} != {want['req_id']}"
    if "error_contains" in want:
        if not reply.is_error:
            return f"expected an error, got {reply}"
        if want["error_contains"] not in reply.error_text():
            return f"error {reply.error_text()!r} lacks {want['error_contains']!r}"
        return None
    if "error_code" in want:
        if not reply.is_error:
            return f"expected an error, got {reply}"
        return None if reply.error_code() == want["error_code"] else f"code {reply.error_code()} != {want['error_code']}"
    if reply.is_error:
        return f"unexpected error {reply.error_text()!r}"
    if "op" in want and want["op"] == "ok" and reply.op != OP_OK:
        return f"want Ok, got op {reply.op:#x}"
    if "value" in want:
        got = decode_value(reply.payload)
        return None if got == as_bytes(want["value"]) else f"value {got!r} != {want['value']!r}"
    if "bool" in want:
        got = bool(reply.payload and reply.payload[0])
        return None if got == want["bool"] else f"bool {got} != {want['bool']}"
    if "mget" in want:
        got = decode_mget(reply.payload)
        exp = [None if v is None else as_bytes(v) for v in want["mget"]]
        return None if got == exp else f"mget {got!r} != {exp!r}"
    if "rows_contain" in want:
        names = decode_vindex_list(reply.payload)
        missing = [n for n in want["rows_contain"] if n not in names]
        return None if not missing else f"vindex list lacks {missing!r} (has {names!r})"
    if "hits_top_id" in want or "hits_len" in want:
        hits = decode_hits(reply.payload)
        if "hits_len" in want and len(hits) != want["hits_len"]:
            return f"want {want['hits_len']} hits, got {len(hits)}"
        if "hits_top_id" in want and (not hits or hits[0][0] != want["hits_top_id"]):
            return f"top hit {hits[:1]!r} != id {want['hits_top_id']}"
        return None
    return None


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
    ap.add_argument("--bin", required=True, help="path to the skeg binary-protocol server")
    ap.add_argument("--cases", default=str(HERE / "native-cases.jsonl"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    port = free_port()
    data_dir = Path(tempfile.mkdtemp(prefix="skeg-native-conformance-"))
    proc = subprocess.Popen(
        [args.bin, "--mode", "rw", "--addr", f"127.0.0.1:{port}", "--data-dir", str(data_dir)],
        env={**os.environ, "RUST_LOG": os.environ.get("RUST_LOG", "warn")},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_tcp(port):
            print("server did not start", file=sys.stderr)
            return 1
        conn = Conn(port)
        failures, bugs = [], []
        for i, case in enumerate(cases, 1):
            want = case["want"]
            try:
                payload = (
                    as_bytes(case["raw_payload"])
                    if "raw_payload" in case
                    else build_payload(case["op"], case.get("args", {}))
                )
                frame = build_frame(case["op"], case.get("version", 1), case.get("req_id", i), payload)
                reply = conn.call(frame)
                reason = check(reply, want, case)
            except Exception as e:  # noqa: BLE001
                reply = None
                # A closed connection is the documented outcome for a frame the
                # server refuses to parse at all.
                if want.get("closed_or_error"):
                    reason = None
                    conn = Conn(port)  # the connection is gone; get a fresh one
                else:
                    reason = f"{type(e).__name__}: {e}"
            if "bug" in case:
                (bugs if reason else failures).append(case["id"])
                print(f"{'bug ' if reason else 'FIXED'} {case['id']}: {case.get('bug', '')}")
                continue
            if reason:
                failures.append((case["id"], reason))
                print(f"FAIL {case['id']}: {reason}")
            elif args.verbose:
                print(f"ok   {case['id']} -> {reply}")

        passed = len(cases) - len(failures) - len(bugs)
        print(f"\n{passed}/{len(cases)} native cases passed, {len(bugs)} known server bug(s)")
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
