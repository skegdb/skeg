#!/usr/bin/env python3
"""Delete only validation objects belonging exclusively to the candidate SHA.

404 is absence. Authentication, permission and transport failures are failures.
A package version also carrying a public tag is never deleted.
"""
import json
import os
import re
import urllib.error
import urllib.request


def main():
    source = os.environ["GITHUB_SHA"]
    assert re.fullmatch(r"[0-9a-f]{40}", source)
    repository = os.environ["GITHUB_REPOSITORY"]
    assert repository == "skegdb/skeg"
    tag = f"validate-{source}"

    def api(path, method="GET", absent=False):
        request = urllib.request.Request("https://api.github.com" + path, method=method,
            headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"],
                     "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
                return json.loads(data) if data else None
        except urllib.error.HTTPError as error:
            try:
                if error.code == 404 and absent:
                    return None
                raise
            finally:
                error.close()

    base = f"/repos/{repository}"
    release = api(f"{base}/releases/tags/{tag}", absent=True)
    ref = api(f"{base}/git/ref/tags/{tag}", absent=True)
    if release:
        assert release["draft"] and release["tag_name"] == tag, "refusing a non-draft release"
    if ref:
        assert ref["object"]["type"] == "commit" and ref["object"]["sha"] == source, "unexpected validation tag target"
    versions = []
    page = 1
    while True:
        batch = api(f"/orgs/skegdb/packages/container/skeg/versions?per_page=100&page={page}", absent=True)
        if not batch:
            break
        for version in batch:
            tags = version["metadata"]["container"]["tags"]
            if tag in tags:
                assert tags == [tag], "scratch manifest also has public tags; refusing deletion"
                versions.append(version["id"])
        page += 1
    # All targets are checked before any deletion.
    if release:
        api(f"{base}/releases/{release['id']}", "DELETE")
    if ref:
        api(f"{base}/git/refs/tags/{tag}", "DELETE")
    for version in versions:
        api(f"/orgs/skegdb/packages/container/skeg/versions/{version}", "DELETE")
    assert api(f"{base}/releases/tags/{tag}", absent=True) is None
    assert api(f"{base}/git/ref/tags/{tag}", absent=True) is None
    for version in versions:
        assert api(f"/orgs/skegdb/packages/container/skeg/versions/{version}", absent=True) is None
    print(json.dumps(dict(source_sha=source, cleanup="verified", package_versions=versions)))


if __name__ == "__main__":
    main()
