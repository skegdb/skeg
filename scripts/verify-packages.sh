#!/usr/bin/env bash
# Test every publishable crate AS IT WILL BE PUBLISHED: packaged, extracted
# outside the workspace, resolved against the registry plus the sibling
# packages produced earlier in publish order - never against the workspace
# and never through its `[patch.crates-io]`, which `cargo package` ignores
# anyway. Catches a crate that only builds thanks to the workspace patch (the
# skeg-multi-tenant / skeg-rigging-skeg engine duplicate), a packaged manifest
# missing files, and a dependency requirement that does not admit the sibling
# version this release puts on the index.
#
# It does not prove: the upload itself, index propagation between publishes,
# or how a third-party consumer's own requirement resolves on the live index.
#
# Usage: scripts/verify-packages.sh [crate ...]   (default: the publish order)
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$PATH"
ORDER=(skeg-proto skeg-simd skeg-platform skeg-telemetry skeg-resp3 skeg-core skeg-vector skeg-tenant skeg-server skeg-server-tenant skeg-multi-tenant)
[ $# -gt 0 ] && ORDER=("$@")
SCRATCH="${VERIFY_SCRATCH:-$(mktemp -d "${TMPDIR:-/tmp}/skeg-verify.XXXXXX")}"
echo "scratch: $SCRATCH"
patch_lines=""
patch_args=()   # the same patches, for `cargo package`, which resolves the registry too
failed=()
for crate in "${ORDER[@]}"; do
  ver=$(cargo metadata --no-deps --format-version 1 | python3 -c "import json,sys; m=json.load(sys.stdin); print(next(p['version'] for p in m['packages'] if p['name']=='$crate'))")
  echo "== $crate $ver"
  # `cargo package` resolves every dependency against the index to write the
  # package's Cargo.lock, even with --no-verify: a sibling bumped in this
  # release but not yet published fails right here - which is exactly what
  # publish-crates would hit if the sibling had not propagated. The siblings
  # extracted earlier in publish order stand in for the index.
  if ! cargo package -p "$crate" --no-verify --allow-dirty -q ${patch_args[@]+"${patch_args[@]}"}; then
    echo "   FAIL: $crate (cargo package)"; failed+=("$crate"); continue
  fi
  dir="$SCRATCH/$crate"
  rm -rf "$dir"; mkdir -p "$dir"
  tar -xzf "target/package/${crate}-${ver}.crate" -C "$dir" --strip-components=1
  mkdir -p "$dir/.cargo"
  # The crate's own tests resolve the crate itself too: a dev-dependency on
  # itself (the failpoints pattern) is rewritten by `cargo package` into a
  # version requirement, which would otherwise pick the PUBLISHED previous
  # version off the index instead of this package. Patch it to this dir
  # first: after the real publish those are the same thing.
  printf '[patch.crates-io]\n%b%s = { path = "%s" }\n' "$patch_lines" "$crate" "$dir" > "$dir/.cargo/config.toml"
  # Packaged crates carry no Cargo.lock we want to trust; resolve fresh, then
  # lock so the test run and a later inspection see the same graph.
  if (cd "$dir" && cargo generate-lockfile -q && cargo test --release -q 2>&1 | tail -n 3); then
    echo "   ok"
  else
    echo "   FAIL: $crate"; failed+=("$crate")
  fi
  (cd "$dir" && cargo tree -e normal -d 2>/dev/null | grep -E '^skeg-' | sed 's/^/   duplicate: /' || true)
  patch_lines="${patch_lines}${crate} = { path = \"${dir}\" }\n"
  # The workspace already patches skeg-platform/simd/vector to its own tree
  # (for skeg-rigging-skeg's registry dependency); a second patch for the
  # same crate at a DIFFERENT path collides, and the manifest patch alone
  # does not cover a member's rewritten path dependency. Same path, no
  # collision: for those three the package step resolves the workspace
  # tree, which is what the extracted package was made from.
  case "$crate" in
    skeg-platform|skeg-simd|skeg-vector) patch_args+=(--config "patch.crates-io.${crate}.path=\"$PWD/crates/${crate}\"") ;;
    *) patch_args+=(--config "patch.crates-io.${crate}.path=\"${dir}\"") ;;
  esac
done
if [ ${#failed[@]} -gt 0 ]; then echo "FAILED: ${failed[*]}"; exit 1; fi
echo "ok: every package builds and tests against the registry plus its published-order siblings"
