"""Destructive cleanup is tested with an in-memory API, never GitHub writes."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

spec = importlib.util.spec_from_file_location("cleanup", Path(__file__).resolve().parents[1] / "cleanup-validation.py")
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.source = "a" * 40
        self.tag = "validate-" + self.source
        self.base = "/repos/skegdb/skeg"
        self.release = dict(id=11, draft=True, tag_name=self.tag)
        self.ref = dict(object=dict(type="commit", sha=self.source))
        self.versions = [dict(id=12, metadata=dict(container=dict(tags=[self.tag])))]
        self.deletes = []
        self.error = 404

    def request(self, req, timeout):
        path = req.full_url.removeprefix("https://api.github.com")
        if req.method == "DELETE":
            self.deletes.append(path)
            if path == self.base + "/releases/11":
                self.release = None
            elif path == self.base + "/git/refs/tags/" + self.tag:
                self.ref = None
            elif path == "/orgs/skegdb/packages/container/skeg/versions/12":
                self.versions = []
            else:
                self.fail("unexpected deletion: " + path)
            return io.BytesIO(b"")
        if path == self.base + "/releases/tags/" + self.tag:
            data = self.release
        elif path == self.base + "/git/ref/tags/" + self.tag:
            data = self.ref
        elif path.endswith("versions?per_page=100&page=1"):
            data = self.versions
        elif "versions?" in path:
            data = []
        elif path.endswith("versions/12"):
            data = next(iter(self.versions), None)
        else:
            self.fail("unexpected read: " + path)
        if data is None:
            raise urllib.error.HTTPError(req.full_url, self.error, "test error", {}, None)
        return io.BytesIO(json.dumps(data).encode())

    def run_cleanup(self):
        with patch.dict(cleanup.os.environ, GITHUB_SHA=self.source,
                        GITHUB_REPOSITORY="skegdb/skeg", GH_TOKEN="not-a-real-token"), \
                patch.object(cleanup.urllib.request, "urlopen", self.request), \
                contextlib.redirect_stdout(io.StringIO()):
            cleanup.main()

    def test_exact_scratch_objects_are_removed_and_verified(self):
        self.run_cleanup()
        self.assertEqual(len(self.deletes), 3)
        self.assertIsNone(self.release)
        self.assertIsNone(self.ref)
        self.assertEqual(self.versions, [])

    def test_cleanup_is_idempotent_when_everything_is_absent(self):
        self.release = self.ref = None
        self.versions = []
        self.run_cleanup()
        self.assertEqual(self.deletes, [])

    def test_public_package_tag_blocks_all_deletions(self):
        self.versions[0]["metadata"]["container"]["tags"].append("latest")
        with self.assertRaises(AssertionError):
            self.run_cleanup()
        self.assertEqual(self.deletes, [])

    def test_wrong_commit_or_published_release_blocks_all_deletions(self):
        for wrong_commit in (True, False):
            self.setUp()
            if wrong_commit:
                self.ref["object"]["sha"] = "b" * 40
            else:
                self.release["draft"] = False
            with self.subTest(wrong_commit=wrong_commit), self.assertRaises(AssertionError):
                self.run_cleanup()
            self.assertEqual(self.deletes, [])

    def test_permission_failure_is_not_absence(self):
        self.release = None
        self.error = 403
        with self.assertRaises(urllib.error.HTTPError):
            self.run_cleanup()
        self.assertEqual(self.deletes, [])
