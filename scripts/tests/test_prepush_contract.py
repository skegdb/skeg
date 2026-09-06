"""Keep the local fixture concurrency consistent with the bounded CI gate."""
from pathlib import Path
import shlex
import unittest


class PrepushContractTests(unittest.TestCase):
    def test_workspace_gate_has_an_explicit_fixture_concurrency_bound(self):
        root = Path(__file__).resolve().parents[2]
        commands = [shlex.split(line) for line in
                    (root / ".githooks/pre-push").read_text().splitlines()
                    if line.startswith("cargo test ")]
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertIn("--all-features", command)
        self.assertIn("--no-fail-fast", command)
        self.assertIn("--", command, "forward a fixture concurrency limit to libtest")
        self.assertIn("--test-threads=1", command[command.index("--") + 1:])
