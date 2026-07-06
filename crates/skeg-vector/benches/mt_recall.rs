#![allow(clippy::cast_precision_loss)]
//! Multitenant FILTERED recall@10 + recall@100 on tq1, via the real RESP path
//! (`search_filtered_hybrid`, IVF-routed). One shared index holds T tenants
//! (M=N/T vectors each, tenant_of(id)=id/M); each query is filtered to one
//! tenant. Ground truth = within-tenant brute-force cosine. Config via env:
//!   SKEG_BENCH_N total, SKEG_MT_TENANTS T, SKEG_TQ1_MODE, SKEG_TQ1_BITPLANE_B,
//!   SKEG_RR rerank, SKEG_CORPUS, SKEG_QUERY, SKEG_DIM, SKEG_NQ.

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};

fn load_npy(path: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(path).unwrap_or_else(|_| panic!("missing {path}"));
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (data, dims[0], dims[1])
}

fn load_prep(path: &str, n_cap: usize, pad: usize) -> Vec<Vec<f32>> {
    let (data, rows, dim) = load_npy(path);
    let n = n_cap.min(rows);
    (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; pad];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() {
    let n_cap: usize = env("SKEG_BENCH_N", 100_000);
    let tenants: usize = env("SKEG_MT_TENANTS", 5);
    let nq: usize = env("SKEG_NQ", 200);
    let native: usize = env("SKEG_DIM", 1024);
    let rerank: usize = env("SKEG_RR", 1600);
    let cpath = std::env::var("SKEG_CORPUS").expect("SKEG_CORPUS");
    let qpath = std::env::var("SKEG_QUERY").expect("SKEG_QUERY");
    let pad = native.next_multiple_of(8);

    let corpus = load_prep(&cpath, n_cap, pad);
    let queries = load_prep(&qpath, nq, pad);
    let n = corpus.len();
    let m = n / tenants; // vectors per tenant
    let n = m * tenants; // trim to a whole multiple

    // SKEG_MT_ISOLATED=1 -> one physical index per tenant (single-tenant walk
    // path); else one shared index + per-tenant FILTER (search_filtered_hybrid).
    let isolated = env("SKEG_MT_ISOLATED", 0u8) == 1;
    let ls: usize = env("SKEG_LS", 1000);

    // Shared index (built only in shared mode).
    let tmp = std::env::temp_dir().join("skeg_mt_tq1");
    let shared_idx = if isolated {
        None
    } else {
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let mut idx = DiskVamanaIndex::create_empty_with_tier(
            &tmp,
            pad,
            300,
            QuantKind::TurboQuant { bits: 1 },
        )
        .unwrap();
        for (id, v) in corpus.iter().take(n).enumerate() {
            idx.insert(id as u64, v).unwrap();
        }
        idx.consolidate().unwrap();
        idx.build_ivf(0, 8).unwrap();
        Some(idx)
    };
    let mode = if isolated {
        "ISOLATED (1 index/tenant, walk)"
    } else {
        "SHARED (1 index + filter)"
    };
    println!(
        "multitenant tq1 [{mode}]: {tenants} tenants x {m} = {n} x {pad}, {} q/tenant, l={ls} rerank={rerank}",
        queries.len()
    );
    println!(
        "{:<8} {:>10} {:>11} {:>8} {:>8} {:>8} {:>9}",
        "tenant", "recall@10", "recall@100", "p50ms", "p99ms", "ramMB", "qps"
    );

    let (mut sum10, mut sum100, mut total_ram) = (0.0f64, 0.0f64, 0.0f64);
    for t in 0..tenants {
        let slice = &corpus[t * m..(t + 1) * m];
        // Per-tenant isolated index (local ids 0..m); else reuse the shared one.
        let iso_tmp = std::env::temp_dir().join(format!("skeg_mt_iso_{t}"));
        let iso_idx = if isolated {
            let _ = std::fs::remove_dir_all(&iso_tmp);
            std::fs::create_dir_all(&iso_tmp).unwrap();
            let mut idx = DiskVamanaIndex::create_empty_with_tier(
                &iso_tmp,
                pad,
                300,
                QuantKind::TurboQuant { bits: 1 },
            )
            .unwrap();
            for (i, v) in slice.iter().enumerate() {
                idx.insert(i as u64, v).unwrap();
            }
            idx.consolidate().unwrap();
            Some(idx)
        } else {
            None
        };
        // id base: isolated uses local ids 0..m; shared uses global t*m..
        let lo = if isolated { 0u64 } else { (t * m) as u64 };
        let ids: Vec<u64> = (lo..lo + m as u64).collect();
        let ram_mb = if isolated {
            iso_idx.as_ref().unwrap().resident_bytes()
        } else {
            shared_idx.as_ref().unwrap().resident_bytes()
        } as f64
            / (1024.0 * 1024.0);
        total_ram += ram_mb;

        let gt: Vec<(AHashSet<u64>, AHashSet<u64>)> = queries
            .par_iter()
            .map(|q| {
                let mut d: Vec<(f32, u64)> = slice
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (cosine_f32(q, v), lo + i as u64))
                    .collect();
                d.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                (
                    d.iter().take(10).map(|&(_, id)| id).collect(),
                    d.iter().take(100).map(|&(_, id)| id).collect(),
                )
            })
            .collect();

        // Search closure: isolated=walk, shared=filtered.
        let run = |q: &[f32], k: usize| -> Vec<(u64, f32)> {
            if isolated {
                iso_idx
                    .as_ref()
                    .unwrap()
                    .search_with_params(q, k, ls, rerank)
                    .unwrap()
            } else {
                shared_idx
                    .as_ref()
                    .unwrap()
                    .search_filtered_hybrid(q, &ids, k, rerank)
                    .unwrap()
            }
        };
        for q in queries.iter().take(3) {
            let _ = run(q, 100);
        }
        let (mut h10, mut h100) = (0usize, 0usize);
        let mut lat = Vec::with_capacity(queries.len());
        for (q, (t10, t100)) in queries.iter().zip(&gt) {
            h10 += run(q, 10).iter().filter(|(id, _)| t10.contains(id)).count();
            let ti = std::time::Instant::now();
            let r100 = run(q, 100);
            lat.push(ti.elapsed().as_secs_f64() * 1e3);
            h100 += r100.iter().filter(|(id, _)| t100.contains(id)).count();
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| lat[(((lat.len() as f64) * p) as usize).min(lat.len() - 1)];
        let jobs: Vec<&Vec<f32>> = (0..10).flat_map(|_| queries.iter()).collect();
        let qt = std::time::Instant::now();
        jobs.par_iter().for_each(|q| {
            let _ = run(q, 100);
        });
        let qps = jobs.len() as f64 / qt.elapsed().as_secs_f64();
        let (r10, r100) = (
            h10 as f64 / (queries.len() * 10) as f64,
            h100 as f64 / (queries.len() * 100) as f64,
        );
        sum10 += r10;
        sum100 += r100;
        println!(
            "t{t:<7} {r10:>10.4} {r100:>11.4} {:>8.2} {:>8.2} {ram_mb:>8.1} {qps:>9.0}",
            pct(0.50),
            pct(0.99)
        );
        drop(iso_idx);
        let _ = std::fs::remove_dir_all(&iso_tmp);
    }
    let ram_note = if isolated {
        format!("{total_ram:.1} (sum)")
    } else {
        format!("{:.1}", total_ram / tenants as f64)
    };
    println!(
        "{:<8} {:>10.4} {:>11.4}  RAM_tot={ram_note}MB",
        "MEAN",
        sum10 / tenants as f64,
        sum100 / tenants as f64
    );
    drop(shared_idx);
    let _ = std::fs::remove_dir_all(&tmp);
}
