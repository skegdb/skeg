#![allow(clippy::cast_precision_loss, clippy::type_complexity)]
//! FilteredVamana increment 2a: on the DISK path (proxy walk + f32 rerank).
//! Builds a DiskVamana (RW insert) with a LABEL-AWARE graph (consolidate_labeled,
//! label = id%100), then compares the single filtered walk vs the qscan and the
//! old two-walk, at 1% / 10%, tq1 & tq2, mxbai. This is the disk+proxy proof
//! (increment 1 was in-memory exact-f32).
//!   SKEG_TIER=tq1|tq2  SKEG_BENCH_N=100000  SKEG_NQ=200

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::PathBuf;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
const CORPUS: &str = "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy";
const QUERY: &str = "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy";
const K: usize = 10;
const RR: usize = 80;

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).ok()?;
    let sh = header.find("'shape':")?;
    let lp = header[sh..].find('(')? + sh + 1;
    let rp = header[lp..].find(')')? + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

fn prep(path: &str, cap: usize) -> (Vec<Vec<f32>>, usize) {
    let (data, rows, dim) = load_npy(path).expect("npy");
    let n = cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = data[i * dim..i * dim + dim].to_vec();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    (out, n)
}

fn main() {
    let n_cap = std::env::var("SKEG_BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let nq = std::env::var("SKEG_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let bits: u8 = if std::env::var("SKEG_TIER").as_deref() == Ok("tq1") {
        1
    } else {
        2
    };
    let (corpus, n) = prep(&format!("{ROOT}/{CORPUS}"), n_cap);
    let (queries, _) = prep(&format!("{ROOT}/{QUERY}"), nq);
    let dim = corpus[0].len();
    println!(
        "disk FilteredVamana: mxbai {n} x {dim}, {} queries, tq{bits}",
        queries.len()
    );

    let tier = QuantKind::TurboQuant { bits };
    let label_of = |id: u64| id % 100; // g = id%100
    let dir: PathBuf = std::env::var("SKEG_STUDY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("skeg_tq1_study"))
        .join(format!("fv_tq{bits}_n{n}"));
    let mut idx = match DiskVamanaIndex::open_with_tier(&dir, tier) {
        Ok(i) if i.len() == n => {
            let mut i = i;
            i.attach_labels(&label_of); // cached label-aware graph, just re-attach
            i
        }
        _ => {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let t = std::time::Instant::now();
            let mut i = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
            for (id, v) in corpus.iter().enumerate() {
                i.insert(id as u64, v).unwrap();
            }
            i.consolidate_labeled(&label_of).unwrap();
            println!("labeled build (RW): {:.0}s", t.elapsed().as_secs_f64());
            i
        }
    };
    idx.attach_labels(&label_of);

    for &thresh in &[1u64, 10] {
        let sel = thresh as f64 / 100.0;
        let ml: Vec<u64> = (0..thresh).collect();
        let matches = move |id: u64| id % 100 < thresh;
        let ids: Vec<u64> = (0..n as u64).filter(|id| id % 100 < thresh).collect();
        let truth: Vec<AHashSet<u64>> = queries
            .par_iter()
            .map(|q| {
                let mut s: Vec<(f32, u64)> = ids
                    .iter()
                    .map(|&id| (cosine_f32(q, &corpus[id as usize]), id))
                    .collect();
                s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                s.iter().take(K).map(|&(_, id)| id).collect()
            })
            .collect();
        println!(
            "-- selectivity {:.0}% ({} matches) --",
            sel * 100.0,
            ids.len()
        );
        let row = |name: &str, f: &dyn Fn(&[f32]) -> Vec<(u64, f32)>| {
            let mut hits = 0usize;
            let t = std::time::Instant::now();
            for (q, tr) in queries.iter().zip(&truth) {
                hits += f(q).iter().filter(|(id, _)| tr.contains(id)).count();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64;
            println!(
                "   {name:<16} recall {:.4}  {ms:.3} ms/q",
                hits as f64 / (queries.len() * K) as f64
            );
        };
        for &l in &[300usize, 600, 1000, 2000] {
            row(&format!("FVamana L{l}"), &|q| {
                idx.search_filtered_labeled(q, K, &ml, l, RR).unwrap()
            });
        }
        row("qscan rr80", &|q| {
            idx.score_ids_quantized(q, &ids, K, RR).unwrap()
        });
        row("old two-walk L300", &|q| {
            idx.search_filtered(q, K, 300, &matches, &[], sel as f32)
                .unwrap()
        });
    }
}
