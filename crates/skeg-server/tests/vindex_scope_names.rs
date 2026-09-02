//! A tenant is not something a client can spell.
//!
//! The vindex map key carries the tenant as a `<32 hex>::<name>` prefix, and
//! five sites in `shard.rs` read the tenant back OUT of that key with
//! `unscope_key` and then act on it - `EraseTenant` picks what to destroy,
//! `warm_payload_indexes` picks which tenant's blobs to read, the blob sweeps
//! pick what to reclaim, `IndexStat` picks the column to report. Every one of
//! them is safe exactly as long as the prefix could only have been put there
//! by the server.
//!
//! It could not: `validate_vindex_name` permits `:`, `scope_key(0, name)`
//! returns the raw name unchanged, and the native protocol always calls with
//! tenant 0 and the client's raw name. So a client on tenant 0 could create
//! `<32 hex of B>::x`, which `unscope_key` then attributes to tenant B -
//! `ERASE TENANT B` destroyed it (and reported zero indexes erased), and B
//! could no longer create its own `x`.
//!
//! The door is closed at CREATE: a raw, client-supplied name containing the
//! scope separator is refused on every entry path. These tests pin the
//! refusal and the two consequences that made it a P1.

use skeg_server::shard::ShardSet;

const DIM: u32 = 8;
/// A tenant id whose 32-hex `to_le_bytes` spelling is what `scope_key` would
/// produce for it. Nothing in this file authenticates as it: the point is that
/// a client on tenant 0 cannot reach it by naming it.
const TENANT_B: u128 = 0x2a00_0000_0000_0000_0000_0000_0000_0001;

/// The map key `scope_key` builds for `(tenant, index)`, spelled out here so
/// the test does not borrow the implementation it is checking.
fn scoped(tenant: u128, index: &str) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    for b in tenant.to_le_bytes() {
        let _ = write!(s, "{b:02x}");
    }
    s.push_str("::");
    s.push_str(index);
    s
}

fn vec_for(id: u64) -> Vec<f32> {
    let mut v = vec![0.05f32; DIM as usize];
    v[(id % 4) as usize] = 1.0;
    v
}

#[tokio::test]
#[ignore = "red until the create door refuses '::' (fix/tenant-carried-not-parsed)"]
async fn a_raw_vindex_name_carrying_the_scope_separator_is_refused() {
    // The ShardSet door: what the native protocol and every admin helper
    // reach. `vindex_create` takes a RAW name here, so `::` in it can only
    // have come from a client.
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), 1).unwrap();
    let squat = scoped(TENANT_B, "x");

    let err = shards
        .vindex_create(&squat, DIM, 4, 1)
        .await
        .expect_err("a raw name containing '::' must be refused");
    assert!(
        err.to_string().contains("must not contain '::'"),
        "the refusal must name the separator, got: {err}"
    );

    let rows = shards.vindex_list().await.unwrap();
    assert!(
        !rows.iter().any(|r| r.name == squat),
        "the refused name must not exist: {rows:?}"
    );
}

#[tokio::test]
#[ignore = "red until the create door refuses '::' (fix/tenant-carried-not-parsed)"]
async fn a_tenant_erase_cannot_be_aimed_by_a_squatted_name() {
    // The P1 in full. Tenant 0 owns a legitimate index; it then tries to
    // create the map key tenant B's own index would use. Before the create
    // door closed, `ERASE TENANT B` walked the registry, saw a key whose
    // `unscope_key` said "B", destroyed tenant 0's index - and reported
    // `Ok((0, 0))`, because the count was taken from the resident map.
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open(dir.path(), 1).unwrap();

    shards.vindex_create("mine", DIM, 4, 1).await.unwrap();
    for id in 0..8u64 {
        shards
            .vset("mine", id, vec_for(id), 0, None, None)
            .await
            .unwrap();
    }

    let squat = scoped(TENANT_B, "x");
    assert!(
        shards.vindex_create(&squat, DIM, 4, 1).await.is_err(),
        "tenant 0 must not be able to create a key that reads as tenant B's"
    );

    let erased = shards
        .erase_tenant(TENANT_B, skeg_core::Durability::Relaxed)
        .await
        .expect("erasing a tenant that owns nothing must still succeed");
    assert_eq!(
        erased,
        (0, 0),
        "tenant B owns nothing here; an erase that counts anything is aimed at \
         someone else's data"
    );

    // The whole point: tenant 0's index survived the erase of a tenant that
    // never existed.
    for id in 0..8u64 {
        let got = shards
            .vget("mine", id)
            .await
            .expect("tenant 0's index must still be there after erasing B");
        assert!(
            got.is_some(),
            "vector {id} of tenant 0's index was destroyed"
        );
    }
}

#[tokio::test]
#[ignore = "red until the create door refuses '::' (fix/tenant-carried-not-parsed)"]
async fn a_registry_key_that_does_not_round_trip_is_an_open_error() {
    // Fail closed on the way out, too. A key that `scope_key(unscope_key(k))`
    // does not reproduce cannot have been written by this server, so nothing
    // downstream can be told which tenant it belongs to. `<32 zeros>::x`
    // unscopes to tenant 0 and index `x`, and re-scoping tenant 0 gives plain
    // `x` - a different key. Silently treating it as tenant 0's `x` is exactly
    // the misattribution the create door exists to prevent, so the open
    // refuses and names the key.
    let dir = tempfile::TempDir::new().unwrap();
    {
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        shards.vindex_create("x", DIM, 4, 1).await.unwrap();
        for id in 0..4u64 {
            shards
                .vset("x", id, vec_for(id), 0, None, None)
                .await
                .unwrap();
        }
        shards.write_snapshot_and_payload_indexes().await;
    }

    let planted = format!("{}::x", "0".repeat(32));
    let shard_dir = dir.path().join("shard-0");
    std::fs::rename(
        shard_dir.join("vindex-x"),
        shard_dir.join(format!("vindex-{planted}")),
    )
    .unwrap();
    rename_registry_entry(&shard_dir.join("vindexes.registry"), "x", &planted);

    let Err(err) = ShardSet::open(dir.path(), 1) else {
        panic!("a registry key that does not round-trip must not open");
    };
    let msg = err.to_string();
    assert!(
        msg.contains(&planted),
        "the error must name the key it refuses, got: {msg}"
    );
}

/// Rewrite one name in a V3 (`SVI3`) registry file, keeping every other field.
/// Byte surgery on purpose: the point is to plant a key the server would never
/// write, which means going around the writer.
fn rename_registry_entry(path: &std::path::Path, from: &str, to: &str) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(
        &bytes[..4],
        b"SVI3",
        "this helper only knows the V3 registry"
    );
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(bytes.len() + to.len());
    out.extend_from_slice(&bytes[..8]);
    let mut pos = 8usize;
    let mut renamed = 0usize;
    for _ in 0..count {
        let nlen = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = std::str::from_utf8(&bytes[pos..pos + nlen]).unwrap();
        pos += nlen;
        let name = if name == from {
            renamed += 1;
            to
        } else {
            name
        };
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(name.len() as u16).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        // dim (4) + tier (1) + generation (16), copied verbatim.
        out.extend_from_slice(&bytes[pos..pos + 21]);
        pos += 21;
    }
    assert_eq!(pos, bytes.len(), "the registry did not parse as V3");
    assert_eq!(renamed, 1, "expected exactly one '{from}' entry to rename");
    std::fs::write(path, out).unwrap();
}
