#!/usr/bin/env python3
"""Stateful release contract on an extracted executable or immutable OCI image.

Fixtures contain PUBLIC test passwords. Only disposable directories are used.
No pip dependencies, rebuilding the artifact, retrying failed tests or demo data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120).stdout.strip()


def container_logs(name):
    result = subprocess.run(["docker", "logs", name], check=True,
                            capture_output=True, text=True, timeout=30)
    return result.stdout + result.stderr


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReplyError(Exception):
    pass


class Conn:
    def __init__(self, port, user=None):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.reader = self.sock.makefile("rb")
        if user:
            try:
                hello = self.cmd("HELLO", 3, "AUTH", user, f"{user}-smoke-only")
                assert isinstance(hello, dict), hello
            except BaseException:
                self.close()
                raise

    def close(self):
        self.reader.close()
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def reply(self):
        line = self.reader.readline()
        if not line.endswith(b"\r\n"):
            raise EOFError("truncated RESP reply")
        kind, value = line[:1], line[1:-2]
        if kind == b"-":
            raise ReplyError(value.decode())
        if kind == b"+":
            return value
        if kind == b":":
            return int(value)
        if kind == b"_":
            return None
        if kind == b"$":
            n = int(value)
            if n == -1:
                return None
            assert 0 <= n <= 64 * 1024 * 1024, "invalid bulk length"
            data = self.reader.read(n)
            assert len(data) == n and self.reader.read(2) == b"\r\n", "truncated bulk"
            return data
        if kind in (b"*", b"%"):
            n = int(value)
            assert 0 <= n <= 4096, "invalid aggregate length"
            if kind == b"%":
                return {self.reply(): self.reply() for _ in range(n)}
            return [self.reply() for _ in range(n)]
        if kind == b",":
            return float(value)
        raise AssertionError(f"unsupported RESP type {kind!r}")

    def cmd(self, *args):
        args = [arg if isinstance(arg, bytes) else str(arg).encode() for arg in args]
        self.sock.sendall(b"*%d\r\n" % len(args) + b"".join(
            b"$%d\r\n" % len(arg) + arg + b"\r\n" for arg in args))
        return self.reply()


def rejected(conn, text, *args):
    try:
        conn.cmd(*args)
    except ReplyError as error:
        assert text.lower() in str(error).lower(), str(error)
        return
    raise AssertionError(f"command {args[0]} unexpectedly succeeded")


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


class Server:
    def __init__(self, root, executable=None, image=None, memory_mib=None):
        self.root, self.executable, self.image = root, executable, image
        self.memory_mib = memory_mib
        self.process, self.container, self.log = None, None, None

    def start(self, readonly=False):
        mode = "serve" if readonly else "rw"
        if self.image:
            self.container = f"skeg-smoke-{uuid.uuid4().hex}"
            mount_mode = "readonly" if readonly else ""
            args = ["docker", "run", "--detach", "--name", self.container,
                    "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=16m",
                    "--pids-limit", "256", "--publish", "127.0.0.1::6379",
                    "--env", "SKEG_SHARDS=1", "--env", "RUST_LOG=info"]
            if self.memory_mib:
                args += [f"--memory={self.memory_mib}m", f"--memory-swap={self.memory_mib}m"]
            for directory in ["data", "auth"]:
                spec = f"type=bind,src={self.root / directory},dst=/{directory}"
                args += ["--mount", spec + (",readonly" if mount_mode else "")]
            args += [self.image, "--addr", "0.0.0.0:6379", "--data-dir", "/data",
                     "--tenant-auth", "/auth/auth.kdb", "--tenant-strict",
                     "--admin-tenant", "admin", "--mode", mode]
            run(*args)
            mapping = run("docker", "port", self.container, "6379/tcp")
            self.port = int(mapping.rsplit(":", 1)[1])
        else:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                self.port = listener.getsockname()[1]
            self.log = tempfile.TemporaryFile()
            self.process = subprocess.Popen([
                str(self.executable), "--addr", f"127.0.0.1:{self.port}",
                "--data-dir", str(self.root / "data"),
                "--tenant-auth", str(self.root / "auth/auth.kdb"), "--tenant-strict",
                "--admin-tenant", "admin", "--mode", mode],
                env=dict(os.environ, SKEG_SHARDS="1", RUST_LOG="warn"),
                stdout=self.log, stderr=self.log, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.container and run("docker", "inspect", "--format", "{{.State.Running}}", self.container) != "true":
                raise AssertionError(container_logs(self.container))
            if self.process and self.process.poll() is not None:
                self.log.seek(0)
                raise AssertionError(self.log.read().decode(errors="replace"))
            try:
                with Conn(self.port) as conn:
                    rejected(conn, "NOAUTH", "HELLO", 3)
                return
            except (ConnectionError, OSError, EOFError):
                # Docker's port forwarder can accept then close before the
                # process has bound. Only a complete NOAUTH reply is readiness.
                # This loop ends before any stateful assertion is attempted.
                time.sleep(0.05)
        raise TimeoutError("server readiness deadline")

    def stop(self, require_success=True):
        if self.container:
            name, self.container = self.container, None
            try:
                run("docker", "stop", "--time", "60", name)
                status = int(run("docker", "inspect", "--format", "{{.State.ExitCode}}", name))
                if require_success:
                    assert status == 0, container_logs(name)
            finally:
                run("docker", "rm", "--force", name)
        if self.process:
            child, self.process = self.process, None
            try:
                child.terminate()
                status = child.wait(timeout=60)
                if require_success:
                    assert status == 0, f"shutdown exit {status}"
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                self.log.close()


def exercise(server):
    vectors = {user: struct.pack("<4f", *values) for user, values in
               [("alice", [1, 2, 3, 4]), ("bob", [4, 3, 2, 1])]}
    server.start()
    with Conn(server.port) as anon:
        rejected(anon, "WRONGPASS", "HELLO", 3, "AUTH", "alice", "incorrect")
    with Conn(server.port, "admin") as admin:
        assert admin.cmd("SKEG.QUOTA.SET", "alice", 2, "*") == b"OK"
    for user, vector in vectors.items():
        with Conn(server.port, user) as conn:
            assert conn.cmd("SET", "shared", user) == b"OK"
            assert conn.cmd("SKEG.VINDEX.CREATE", "shared-index", 4, "disk") == b"OK"
            assert conn.cmd("SKEG.VSET", "shared-index", 11, vector) == b"OK"
            assert conn.cmd("SKEG.VSET", "shared-index", 12, vector) == b"OK"
            if user == "alice":
                rejected(conn, "quota", "SKEG.VSET", "shared-index", 13, vector)
    server.stop()
    server.start()
    check_reads(server, vectors, readonly=False)
    server.stop()
    before = inventory(server.root)
    # OCI gets real read-only bind mounts. For a native tarball also remove
    # write permission; the inventory assertion remains mandatory on both.
    for directory in [server.root / "data", server.root / "auth"]:
        for path in directory.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        directory.chmod(0o555)
    try:
        server.start(readonly=True)
        check_reads(server, vectors, readonly=True)
        server.stop()
        after = inventory(server.root)
        assert before == after, "serve changed persisted bytes or file inventory"
        return before
    finally:
        for directory in [server.root / "data", server.root / "auth"]:
            directory.chmod(0o777)
            for path in directory.rglob("*"):
                path.chmod(0o777 if path.is_dir() else 0o666)


def check_reads(server, vectors, readonly):
    with Conn(server.port, "admin") as admin:
        assert admin.cmd("SKEG.QUOTA.GET", "alice") == [b"2", b"*"]
    for user, vector in vectors.items():
        with Conn(server.port, user) as conn:
            assert conn.cmd("GET", "shared") == user.encode(), "KV tenant/restart mismatch"
            assert conn.cmd("SKEG.VGET", "shared-index", 11) == vector, "vector tenant/restart mismatch"
            if readonly:
                rejected(conn, "read", "SET", "shared", "forbidden")
                rejected(conn, "read", "SKEG.VSET", "shared-index", 11, vector)
            elif user == "alice":
                rejected(conn, "quota", "SKEG.VSET", "shared-index", 13, vector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tarball", type=Path)
    source.add_argument("--image")
    source.add_argument("--binary", type=Path, help="development only; not release artifact proof")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--source-sha", default=run("git", "-C", str(ROOT), "rev-parse", "HEAD"))
    args = parser.parse_args()
    assert re.fullmatch(r"[0-9a-f]{40}", args.source_sha), "full source SHA required"
    actual_sha = run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    assert args.source_sha == actual_sha, "source SHA differs from checked out validator"
    evidence = dict(source_sha=args.source_sha, status="failed", platform=os.uname().machine,
                    source_dirty=bool(run("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")),
                    artifact_kind="image" if args.image else "tarball" if args.tarball else "development")
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.require_clean:
            assert not evidence["source_dirty"], "release smoke requires clean tracked sources"
        with tempfile.TemporaryDirectory(prefix="skeg-artifact-") as scratch:
            root = Path(scratch)
            for directory in ["data", "auth"]:
                (root / directory).mkdir(mode=0o777)
                (root / directory).chmod(0o777)
            auth = root / "auth/auth.kdb"
            auth.write_bytes(bytes.fromhex((ROOT / "scripts/fixtures/artifact-auth.hex").read_text()))
            auth.chmod(0o444)
            executable, image = args.binary, None
            if args.tarball:
                evidence["tarball_sha256"] = sha(args.tarball)
                expected = Path(str(args.tarball) + ".sha256").read_text().split()[0]
                assert expected == evidence["tarball_sha256"], "tarball checksum mismatch"
                extracted = root / "extracted"
                extracted.mkdir()
                with tarfile.open(args.tarball) as archive:
                    assert sorted(m.name for m in archive.getmembers()) == ["skeg", "skeg-resp3"], "unexpected/duplicate tar members"
                    for member in archive.getmembers():
                        assert member.name in ("skeg", "skeg-resp3") and member.isfile(), "unexpected tar member"
                        with archive.extractfile(member) as src:
                            (extracted / member.name).write_bytes(src.read())
                        (extracted / member.name).chmod(0o755)
                assert (extracted / "skeg").is_file(), "missing native binary"
                run(str(extracted / "skeg"), "--help")
                evidence["native_executable_sha256"] = sha(extracted / "skeg")
                executable = extracted / "skeg-resp3"
            if args.image:
                metadata = json.loads(run("docker", "image", "inspect", args.image))[0]
                assert metadata["Config"]["Entrypoint"] == ["/usr/local/bin/skeg-resp3"]
                image = metadata["Id"]
                if args.require_clean:
                    assert "@sha256:" in args.image, "published OCI smoke requires a digest reference"
                    assert metadata["Config"].get("Labels", {}).get("org.opencontainers.image.revision") == args.source_sha
                evidence.update(image_id=image, oci_digests=metadata.get("RepoDigests", []))
                evidence["executable_sha256"] = run("docker", "run", "--rm", "--entrypoint",
                    "sha256sum", image, "/usr/local/bin/skeg-resp3").split()[0]
            else:
                executable = executable.resolve()
                evidence["executable_sha256"] = sha(executable)
                help_text = run(str(executable), "--help")
                assert "--tenant-strict" in help_text and "--tenant-auth" in help_text
            server = Server(root, executable, image)
            try:
                evidence["inventory"] = exercise(server)
            except Exception:
                if server.container:
                    evidence["container_state"] = json.loads(run("docker", "inspect", "--format", "{{json .State}}", server.container))
                    evidence["container_log"] = container_logs(server.container)
                raise
            finally:
                server.stop(require_success=False)
            evidence["status"] = "passed"
    except Exception as error:
        evidence["error"] = str(error)
        raise
    finally:
        args.evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"PASS stateful artifact smoke: {args.evidence}")


if __name__ == "__main__":
    main()
