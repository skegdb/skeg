//! Tenant erasure against real embeddings, in a real multi-tenant shape.
//!
//! The synthetic sweep (`erase_tenant_scaling`) answers "what does the walk
//! cost". This one answers the question that actually decides whether the
//! feature is shippable: **when one tenant is erased, does everyone else come
//! through untouched, and is the erased tenant really gone?**
//!
//! Shape, per embedding in the roster: `TENANTS` tenants share the shards. Each
//! owns a vindex over its own slice of a real corpus, a payload blob per vector
//! (which is itself a KV key under the tenant's prefix, so the erase path has to
//! reclaim it), and a handful of plain KV docs. We measure a survivor's
//! recall@10 and search latency, erase one tenant, then measure the survivor
//! again on the identical queries.
//!
//! What the numbers have to say:
//!   - `erase ms`      : what the sweep costs with vectors and blobs in play.
//!   - `recall drift`  : survivor recall after minus before. Must be 0.000.
//!   - `p99 drift`     : survivor tail after minus before. Must be ~0.
//!   - `victim keys`   : keys left behind by the erased tenant. Must be 0.
//!   - `victim disk`   : bytes still charged to them. Must be 0.
//!
//! Recall is against an exact brute-force ground truth over the survivor's own
//! slice, so a survivor whose index got clipped by the neighbour's erasure shows
//! up as a recall drop, not as a silent pass.
//!
//! `harness = false`, custom main, markdown table to stdout. Knobs (env):
//!   SKEG_TENANTS (4)  SKEG_N (20000, vectors per tenant)  SKEG_NQ (200)
//!   SKEG_LS (200)     SKEG_BITS (2 -> tq2, the RW default)
//!   SKEG_EMBEDS (comma-separated roster names, default: all found in hf_data)

use std::time::Instant;

use bytes::Bytes;
use skeg_core::group_commit::Durability;
use skeg_server::shard::ShardSet;
use tempfile::TempDir;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const SHARDS: usize = 4;
const K: usize = 10;

/// The real-embedding roster: `(name, file stem, native dim)`.
const ROSTER: &[(&str, &str, usize)] = &[
    ("glove-104", "glove104", 104),
    ("instructor-xl", "arxiv-instructorxl", 768),
    ("gemini-001", "gemini-001", 768),
    ("cohere-wiki", "wiki-cohere", 1024),
    ("openai-3-large", "openai3-large", 1536),
];

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

/// Read a float32 `.npy`, truncate to `cap` rows, zero-pad each row to `pad`,
/// and L2-normalise (cosine == dot on the unit sphere).
fn load_npy(path: &str, cap: usize, pad: usize) -> (Vec<f32>, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("read {path}"));
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, native) = (dims[0].min(cap), dims[1]);
    let start = 10 + hlen;
    let mut out = vec![0f32; rows * pad];
    for r in 0..rows {
        for c in 0..native {
            let off = start + (r * native + c) * 4;
            out[r * pad + c] =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
    }
    for row in out.chunks_exact_mut(pad) {
        let nrm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in row.iter_mut() {
            *x /= nrm;
        }
    }
    (out, rows)
}

/// Exact top-`K` over `corpus[base .. base+n]` by dot product, returned as the
/// ids the server would use (`base + row`).
fn brute_force(corpus: &[f32], base: usize, n: usize, q: &[f32], dim: usize) -> Vec<u64> {
    let mut scored: Vec<(f32, u64)> = (0..n)
        .map(|r| {
            let row = &corpus[(base + r) * dim..(base + r + 1) * dim];
            let dot: f32 = row.iter().zip(q).map(|(a, b)| a * b).sum();
            (dot, (base + r) as u64)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(K).map(|(_, id)| id).collect()
}

fn scoped_doc(tenant: u128, i: usize) -> Vec<u8> {
    let mut k = tenant.to_le_bytes().to_vec();
    k.extend_from_slice(format!("doc:{i}").as_bytes());
    k
}

fn vindex_name(tenant: u128) -> String {
    // Mirrors shard::scope_key: 32 lowercase hex of the tenant's LE bytes.
    let mut s = String::with_capacity(38);
    for b in tenant.to_le_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s.push_str("::idx");
    s
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

/// Search the survivor's index on every query; return `(recall@10, p50_ms, p99_ms)`.
async fn probe(
    shards: &ShardSet,
    tenant: u128,
    queries: &[f32],
    nq: usize,
    dim: usize,
    l_search: u32,
    truth: &[Vec<u64>],
) -> (f64, f64, f64) {
    let name = vindex_name(tenant);
    let mut hits = 0usize;
    let mut lat = Vec::with_capacity(nq);
    for (qi, want) in truth.iter().enumerate().take(nq) {
        let q = queries[qi * dim..(qi + 1) * dim].to_vec();
        let t0 = Instant::now();
        let got = shards
            .vsearch(&name, q, K, l_search, tenant, false, None)
            .await
            .unwrap();
        lat.push(t0.elapsed().as_secs_f64() * 1e3);
        hits += got.iter().filter(|(id, ..)| want.contains(id)).count();
    }
    lat.sort_unstable_by(f64::total_cmp);
    let recall = hits as f64 / (nq * K) as f64;
    (recall, pct(&lat, 0.50), pct(&lat, 0.99))
}

#[allow(clippy::too_many_lines)]
fn main() {
    let tenants: u128 = env("SKEG_TENANTS", 4u128);
    let n: usize = env("SKEG_N", 20_000);
    let nq: usize = env("SKEG_NQ", 200);
    let l_search: u32 = env("SKEG_LS", 200);
    let bits: u8 = env("SKEG_BITS", 2);
    let kind: u8 = match bits {
        1 => 3,
        2 => 4,
        4 => 5,
        other => panic!("SKEG_BITS must be 1, 2 or 4 (got {other})"),
    };
    let only: Option<Vec<String>> = std::env::var("SKEG_EMBEDS")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_owned()).collect());

    assert!(tenants >= 2, "need a victim and at least one survivor");
    // Tenant ids are 1..=tenants; 0 is the anonymous tenant and cannot be erased.
    let victim: u128 = 1;
    let survivor: u128 = 2;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("# Tenant erasure on real embeddings");
    println!(
        "\n{tenants} tenants x {n} vectors each (tq{bits}, {SHARDS} shards, l_search={l_search}, \
         {nq} queries).\nTenant 1 is erased; tenant 2 is measured before and after. Drifts must be zero.\n"
    );
    println!(
        "| embedding | dim | erase ms | reclaim ms | MiB freed | keys erased | vindexes | recall@10 before | after | drift | p99 ms before | after | victim keys left | victim disk left |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");

    for &(label, stem, native) in ROSTER {
        if let Some(only) = &only
            && !only.iter().any(|s| s == label)
        {
            continue;
        }
        let cpath = format!("{ROOT}/hf_data/{stem}_corpus.npy");
        let qpath = format!("{ROOT}/hf_data/{stem}_queries.npy");
        if !std::path::Path::new(&cpath).exists() {
            println!("| {label} | - | _corpus missing, skipped_ | | | | | | | | | |");
            continue;
        }
        let dim = native.next_multiple_of(8);
        let need = n * tenants as usize;
        let (corpus, rows) = load_npy(&cpath, need, dim);
        if rows < need {
            println!(
                "| {label} | {native} | _corpus has {rows} rows, need {need}, skipped_ | | | | | | | | | |"
            );
            continue;
        }
        let (queries, nq_rows) = load_npy(&qpath, nq, dim);
        let nq = nq.min(nq_rows);

        let dir = TempDir::new().unwrap();
        let row = rt.block_on(async {
            let shards = ShardSet::open(dir.path(), SHARDS).unwrap();

            // Each tenant owns a distinct slice of the corpus: ids are global row
            // numbers, so a leak across tenants would show up as a foreign id.
            for t in 1..=tenants {
                let base = (t as usize - 1) * n;
                let name = vindex_name(t);
                shards
                    .vindex_create(&name, dim as u32, kind, 0)
                    .await
                    .unwrap();
                for r in 0..n {
                    let id = (base + r) as u64;
                    let vec = corpus[(base + r) * dim..(base + r + 1) * dim].to_vec();
                    shards
                        .vset(
                            &name,
                            id,
                            vec,
                            t,
                            None,
                            // A payload blob per vector: a KV key under the
                            // tenant's prefix, which the erase path must reclaim.
                            Some(Bytes::from(format!("payload:{id}"))),
                        )
                        .await
                        .unwrap();
                }
                // Plain KV docs alongside the vectors.
                for i in 0..64 {
                    shards
                        .set(&scoped_doc(t, i), b"v", Durability::Relaxed)
                        .await
                        .unwrap();
                }
            }

            // Ground truth for the survivor, over the survivor's slice only.
            let sbase = (survivor as usize - 1) * n;
            let truth: Vec<Vec<u64>> = (0..nq)
                .map(|qi| brute_force(&corpus, sbase, n, &queries[qi * dim..(qi + 1) * dim], dim))
                .collect();

            let (r_before, _p50_b, p99_before) =
                probe(&shards, survivor, &queries, nq, dim, l_search, &truth).await;

            let t0 = Instant::now();
            let (vindexes, keys) = shards
                .erase_tenant(victim, Durability::Relaxed)
                .await
                .unwrap();
            let erase_ms = t0.elapsed().as_secs_f64() * 1e3;

            // Physical reclaim: erase only tombstoned; this compacts the dead
            // bytes off disk. Timed and measured (bytes freed) on real data.
            let t1 = Instant::now();
            let freed = shards.reclaim().await.unwrap();
            let reclaim_ms = t1.elapsed().as_secs_f64() * 1e3;

            let (r_after, _p50_a, p99_after) =
                probe(&shards, survivor, &queries, nq, dim, l_search, &truth).await;

            // What did the victim leave behind? Their vector payload blobs and
            // KV docs all carry their prefix, so a survivor of the sweep is a
            // leak. Counted the same way the sweep found them.
            let victim_left = shards.count_tenant_keys(victim).await.unwrap();
            let victim_disk = shards.tenant_disk_bytes(victim);

            (
                erase_ms,
                reclaim_ms,
                freed,
                keys,
                vindexes,
                r_before,
                r_after,
                p99_before,
                p99_after,
                victim_left,
                victim_disk,
            )
        });

        let (ms, rec_ms, freed, keys, vidx, rb, ra, p99b, p99a, left, disk) = row;
        let freed_mib = freed as f64 / (1024.0 * 1024.0);
        println!(
            "| {label} | {native} | {ms:.1} | {rec_ms:.1} | {freed_mib:.1} | {keys} | {vidx} | {rb:.3} | {ra:.3} | {:+.3} | {p99b:.2} | {p99a:.2} | {left} | {disk} |",
            ra - rb
        );
    }
}
