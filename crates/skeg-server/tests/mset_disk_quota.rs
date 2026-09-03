//! `MSET` never enforced `max_disk_bytes` at all - a deterministic full
//! bypass (audit/17 round 2), found after `SET`/`APPEND`/the payload blob
//! path were already closed. `MSET` shares `VLog::set_many` with nothing
//! else, and `set_many` never took a limit.
//!
//! `ShardSet::mset_with_disk_limit`/`VLog::set_many_with_disk_limit` close
//! it the same way: the batch's SUM (its net delta to the tenant's charge,
//! not a per-pair check) is checked and reserved atomically, in the same
//! critical section `set_scoped` uses, before a single byte is written -
//! `set_many`'s own all-or-nothing contract extended to the disk quota
//! rather than stopped short of it. See `docs/adr-payload-transaction.md`,
//! "Disk quota".

use skeg_server::admission::AdmissionError;
use skeg_server::shard::{ShardError, ShardSet};

/// One shard: `mset` is only atomic PER SHARD (documented, pre-existing,
/// unrelated to this fix - a batch split across shards is several
/// independent commits with no cross-shard coordination). A single shard is
/// what makes "the whole batch" and "one shard's portion" the same thing,
/// which is the all-or-nothing property these tests are pinning.
fn open(dir: &std::path::Path) -> ShardSet {
    ShardSet::open(dir, 1).expect("shard set opens")
}

fn scoped_key(tenant: u128, raw: &[u8]) -> Vec<u8> {
    let mut k = tenant.to_le_bytes().to_vec();
    k.extend_from_slice(raw);
    k
}

/// A batch whose sum would cross the tenant's limit writes NONE of its
/// members and reports the typed refusal, not a Storage string.
#[tokio::test]
#[ignore = "opens in 'core: enforce the tenant disk limit on set_many'"]
async fn an_mset_that_would_exceed_the_limit_is_refused_admission_typed_and_writes_nothing() {
    const T: u128 = 0x3001;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());

    // Probe: what does one of these pairs cost? Then undo it.
    let ka = scoped_key(T, b"a");
    shards
        .tenant(T)
        .with_disk_limit(None)
        .set(&ka, &vec![0u8; 512], skeg_core::Durability::Kernel)
        .await
        .unwrap();
    let unit = shards.tenant_disk_bytes(T);
    assert!(
        shards
            .del(&ka, skeg_core::Durability::Kernel)
            .await
            .unwrap()
    );
    assert_eq!(shards.tenant_disk_bytes(T), 0);

    // Room for 1.5 units; the batch below asks for 2.
    let limit = unit + unit / 2;
    let kb = scoped_key(T, b"b");
    let big = vec![0u8; 512];
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (ka.as_slice(), big.as_slice()),
        (kb.as_slice(), big.as_slice()),
    ];

    let err = shards
        .mset_with_disk_limit(&pairs, skeg_core::Durability::Kernel, T, Some(limit))
        .await
        .expect_err("a batch whose sum exceeds the limit must be refused");

    match err {
        ShardError::Admission(AdmissionError::DiskQuota {
            tenant, limit: l, ..
        }) => {
            assert_eq!(tenant, T);
            assert_eq!(l, limit);
        }
        other => panic!(
            "an MSET disk-quota refusal must classify as \
             ShardError::Admission(AdmissionError::DiskQuota {{ .. }}), not {other:?}"
        ),
    }
    assert_eq!(
        shards.tenant_disk_bytes(T),
        0,
        "a refused batch must not move the counter"
    );
    assert_eq!(
        shards.get(&ka).await.unwrap(),
        None,
        "neither key was written"
    );
    assert_eq!(
        shards.get(&kb).await.unwrap(),
        None,
        "neither key was written"
    );
}

/// A batch that fits is admitted whole, and the counter equals exactly the
/// sum of what it wrote.
#[tokio::test]
async fn an_mset_that_fits_is_admitted_and_counted() {
    const T: u128 = 0x3002;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());
    let ka = scoped_key(T, b"a");
    let kb = scoped_key(T, b"b");
    let va = b"aaaa".to_vec();
    let vb = b"bbbb".to_vec();
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (ka.as_slice(), va.as_slice()),
        (kb.as_slice(), vb.as_slice()),
    ];

    shards
        .mset_with_disk_limit(&pairs, skeg_core::Durability::Kernel, T, Some(1 << 20))
        .await
        .expect("a batch well under budget must be admitted");

    assert_eq!(
        shards.get(&ka).await.unwrap().as_deref(),
        Some(va.as_slice())
    );
    assert_eq!(
        shards.get(&kb).await.unwrap().as_deref(),
        Some(vb.as_slice())
    );
    assert!(shards.tenant_disk_bytes(T) > 0);
}

/// Two tenants on one disk: A pinned at its own limit cannot use the MSET
/// path to touch B's budget, and B (unlimited) writes freely.
#[tokio::test]
#[ignore = "opens in 'core: enforce the tenant disk limit on set_many'"]
async fn two_tenants_one_disk_mset_a_at_its_limit_cannot_touch_b() {
    const A: u128 = 0x3003;
    const B: u128 = 0x3004;
    let dir = tempfile::TempDir::new().unwrap();
    let shards = open(dir.path());

    let ka = scoped_key(A, b"a");
    shards
        .tenant(A)
        .with_disk_limit(None)
        .set(&ka, b"xxxx", skeg_core::Durability::Kernel)
        .await
        .unwrap();
    let a_used = shards.tenant_disk_bytes(A);
    assert!(a_used > 0);

    let ka2 = scoped_key(A, b"a2");
    let big = vec![0u8; 4096];
    let refused_pairs: Vec<(&[u8], &[u8])> = vec![(ka2.as_slice(), big.as_slice())];
    assert!(
        shards
            .mset_with_disk_limit(
                &refused_pairs,
                skeg_core::Durability::Kernel,
                A,
                Some(a_used)
            )
            .await
            .is_err(),
        "A must be refused at its own limit"
    );

    let kb = scoped_key(B, b"b");
    let ok_pairs: Vec<(&[u8], &[u8])> = vec![(kb.as_slice(), big.as_slice())];
    shards
        .mset_with_disk_limit(&ok_pairs, skeg_core::Durability::Kernel, B, None)
        .await
        .expect("B has its own budget and A's limit must not apply to it");

    assert!(shards.tenant_disk_bytes(B) > 0);
    assert_eq!(
        shards.tenant_disk_bytes(A),
        a_used,
        "B's write must not have moved A's counter"
    );
}
