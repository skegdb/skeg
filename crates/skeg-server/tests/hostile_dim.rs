//! A dimension a client chooses must not be able to kill the process.
//!
//! `SKEG.VINDEX.CREATE name dim [kind] backend` parses `dim` as a plain u32
//! and the default tier is 2-bit TurboQuant, whose packing asserts
//! `dim % 4 == 0`. The crate builds with `panic = "abort"`, so an assertion
//! is not an error a client sees - it is the process, for every tenant.
//!
//! Run as a subprocess so an abort is observable rather than fatal to the
//! test runner.

use skeg_server::shard::ShardSet;
use skeg_vector::QuantKind;

#[tokio::test]
async fn a_dimension_the_quantiser_cannot_pack_is_refused_not_fatal() {
    let dir = tempfile::TempDir::new().unwrap();
    let shards = ShardSet::open_mode_with_workers(
        dir.path(),
        1,
        false,
        QuantKind::TurboQuant { bits: 2 },
        1,
    )
    .unwrap();

    // Wire kind 4 IS the 2-bit TurboQuant a client gets by default
    // (`DEFAULT_KIND_TQ2 = 4` in the RESP handler - the byte is not the bit
    // count, and reading it as one is how the first version of this test
    // measured a different tier than it thought). Backend 1 = disk.
    for dim in [1u32, 2, 3, 5, 7, 17, 31, 33, 1023, 1024] {
        let name = format!("d{dim}");
        let created = shards.vindex_create(&name, dim, 4, 1).await;
        match created {
            // Refusing is the correct answer.
            Err(e) => {
                let msg = format!("{e}");
                eprintln!("dim {dim}: RIFIUTATO -> {msg}");
                assert!(
                    msg.contains("dim") || msg.contains("divis") || msg.contains("packing"),
                    "dim {dim} was refused, but not for a reason a client can act on: {msg}"
                );
            }
            // Accepting it means the index must actually work.
            Ok(()) => {
                let kind_now = shards
                    .vindex_list()
                    .await
                    .unwrap()
                    .into_iter()
                    .find(|r| r.name == name)
                    .map(|r| r.kind);
                eprintln!("dim {dim}: ACCETTATO, kind = {kind_now:?} (4 = tq2 richiesto)");
                let v = vec![0.5f32; dim as usize];
                shards
                    .vset(&name, 1, v.clone(), 0, None, None)
                    .await
                    .unwrap_or_else(|e| panic!("dim {dim} was accepted but VSET failed: {e}"));
                let got = shards.vget(&name, 1).await.unwrap();
                assert!(got.is_some(), "dim {dim} accepted, stored, and lost");
                // And it must survive MAINTENANCE, which is where the codes are
                // actually packed. A VSET only reaches the delta, so an index
                // with an unpackable dim looks healthy right up to the first
                // fold - and that runs on the maintenance loop, not on the
                // request that caused it.
                for id in 2..64u64 {
                    shards
                        .vset(&name, id, vec![0.25f32; dim as usize], 0, None, None)
                        .await
                        .unwrap();
                }
                shards
                    .vindex_consolidate(&name)
                    .await
                    .unwrap_or_else(|e| panic!("dim {dim}: the fold failed: {e}"));
                assert!(
                    shards.vget(&name, 1).await.unwrap().is_some(),
                    "dim {dim}: row lost across the fold"
                );
            }
        }
    }
}
