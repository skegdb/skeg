import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect = load("collect-release-evidence").collect
smoke = load("smoke-resp3-artifact")
counters = load("cgroup-evidence")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.source = "a" * 40
        self.names = [f"tarball-{t}.json" for t in (
            "aarch64-apple-darwin", "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu")]
        self.names += ["image-0.json", "image-1.json"]
        self.proof = dict(status="passed", source_sha=self.source, source_dirty=False,
                          executable_sha256="b" * 64, native_executable_sha256="c" * 64,
                          tarball_sha256="d" * 64, oci_digests=["repo@sha256:" + "e" * 64],
                          inventory={"data/file": "f" * 64})
        for name in self.names:
            (self.path / name).write_text(json.dumps(self.proof))

    def test_complete_proof_is_collected(self):
        self.assertEqual(len(collect(self.path, self.source)["artifacts"]), 5)

    def test_missing_arm_proof_blocks_promotion(self):
        (self.path / self.names[1]).unlink()
        with self.assertRaises(FileNotFoundError):
            collect(self.path, self.source)

    def test_stale_dirty_failed_and_digestless_proofs_block_promotion(self):
        for field, value in [("source_sha", "0" * 40), ("source_dirty", True),
                             ("status", "failed"), ("oci_digests", []), ("inventory", {})]:
            with self.subTest(field=field):
                proof = dict(self.proof, **{field: value})
                (self.path / "image-0.json").write_text(json.dumps(proof))
                with self.assertRaises(AssertionError):
                    collect(self.path, self.source)

    def test_truncated_bulk_cannot_be_accepted_as_a_valid_value(self):
        conn = smoke.Conn.__new__(smoke.Conn)
        conn.reader = io.BytesIO(b"$5\r\nab\r\n")
        with self.assertRaises(AssertionError):
            conn.reply()

    def test_complete_map_leaves_the_next_reply_intact(self):
        conn = smoke.Conn.__new__(smoke.Conn)
        conn.reader = io.BytesIO(b"%1\r\n+version\r\n$3\r\n0.8\r\n+OK\r\n")
        self.assertEqual(conn.reply(), {b"version": b"0.8"})
        self.assertEqual(conn.reply(), b"OK")

    def test_kernel_counters_require_real_limit_and_no_oom(self):
        text = "memory.max:\n268435456\nmemory.peak:\n123456\nmemory.events:\nlow 0\nmax 0\noom 0\noom_kill 0\n"
        self.assertEqual(counters.validate(text, 256)["memory_peak_bytes"], 123456)
        for bad in [text.replace("oom_kill 0", "oom_kill 1"), text.replace("oom 0", "oom 1"),
                    text.replace("268435456", "536870912"), text.replace("memory.peak:", "missing:")]:
            with self.subTest(text=bad), self.assertRaises(AssertionError):
                counters.validate(bad, 256)


if __name__ == "__main__":
    unittest.main()
