//! Fase 0.c (ram-reduction) - gate recall PQ come tier di navigazione.
//!
//! Not a Criterion bench: a reporting harness (`harness = false`).
//!
//! design-pq-tier.md: Product Quantization (codebook k-means, M sotto-vettori,
//! K centroidi) come tier al posto dell'int8. 16x piu' piccolo. A differenza
//! di RaBitQ/binary/4-bit (gia' falsificati) PQ e' data-dependent.
//!
//! Il gate critico: PQ non deve solo *rankare*, deve *navigare* il greedy
//! walk del grafo - la barra che ha ucciso binary (0.748) e 4-bit (0.689).
//! Il walk e' guidato dal proxy PQ, poi re-rank f32, recall@10 vs brute
//! force. Confronto: f32 (soffitto) / int8 (baseline) / PQ.
//!
//! Dataset: mxbai reale 10K (ancora reale, il gate di design-pq-tier.md) +
//! sintetico a manifold a 100K/1M per la degradazione di scala. uniform-sphere
//! a 1024-dim NON e' usabile: a scala non e' indicizzabile (il soffitto f32
//! stesso crolla), confonderebbe il gate.
//!
//! Gate: recall@10 PQ >= 0.98 (relativo: PQ deve restare vicino a int8).

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;
use skeg_vector::{DiskVamanaIndex, VamanaConfig, VamanaIndex};

const CORPUS_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-embed-large_10000.npy"
);
const QUERY_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-embed-large_200.npy"
);
// Real Simple-English-Wikipedia passages embedded via mxbai-embed-large: the
// clean real-document scaling anchor (the 10K corpus above is word-salad text;
// the synthetic manifold has a near-tie confound at scale).
const WIKI_CORPUS_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/corpus_mxbai-wiki.npy"
);
const WIKI_QUERY_NPY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy"
);
const K: usize = 10; // recall@10
const L_SEARCH: usize = 100; // beam, matches VamanaConfig::default().l_search
const TRAIN_SAMPLE: usize = 50_000; // codebook training sample cap

fn load_npy(path: &str) -> Option<(Vec<f32>, usize, usize)> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return None;
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).ok()?;
    let sh = header.find("'shape':")?;
    let lp = header[sh..].find('(')? + sh + 1;
    let rp = header[lp..].find(')')? + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if dims.len() != 2 {
        return None;
    }
    let data: Vec<f32> = bytes[10 + header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Some((data, dims[0], dims[1]))
}

/// Cosine similarity of two equal-length vectors.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

/// Unit-normalised copy of `v`.
fn normalized(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// Squared L2 distance.
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// xorshift64 -> f32 in [0,1).
fn rand_f32(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f32 / (1u64 << 53) as f32
}

/// One standard-normal sample (Box-Muller).
fn gaussian(state: &mut u64) -> f32 {
    let u1 = rand_f32(state).max(1e-9);
    let u2 = rand_f32(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// A fixed random `d_in` x `dim` projection matrix.
fn manifold_proj(dim: usize, d_in: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..d_in * dim).map(|_| gaussian(&mut s)).collect()
}

/// `n` vectors on the `d_in`-dimensional manifold defined by `proj`: a random
/// `d_in`-gaussian projected through `proj`, plus small isotropic noise, then
/// unit-normalised. Corpus and queries must share the same `proj` to live on
/// the same manifold. Models real embeddings (low intrinsic dimension,
/// indexable) far better than a full-dim uniform sphere.
fn manifold_points(proj: &[f32], n: usize, dim: usize, d_in: usize, seed: u64) -> Vec<f32> {
    let noise = 0.05f32;
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut st = (seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_add(i as u64 * 2_654_435_761);
            st |= 1;
            let z: Vec<f32> = (0..d_in).map(|_| gaussian(&mut st)).collect();
            let mut v = vec![0.0f32; dim];
            for d in 0..dim {
                let mut acc = 0.0f32;
                for j in 0..d_in {
                    acc += z[j] * proj[j * dim + d];
                }
                v[d] = acc + noise * gaussian(&mut st);
            }
            normalized(&v)
        })
        .collect()
}

// ── PQ codebook ───────────────────────────────────────────────────────────────

/// PQ codebook: `m` subvectors, `k` centroids each, subvector dim `sub_dim`.
struct PqCodebook {
    m: usize,
    k: usize,
    sub_dim: usize,
    centroids: Vec<Vec<f32>>, // centroids[s] = flat k * sub_dim
}

/// Lloyd's k-means on `points` (flat, `n` rows of `dim`), random init,
/// `iters` iterations. Empty clusters reseed to a random point.
fn kmeans(points: &[f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut centroids = vec![0.0f32; k * dim];
    for c in 0..k {
        let r = (next() as usize) % n;
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&points[r * dim..(r + 1) * dim]);
    }
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        assign.par_iter_mut().enumerate().for_each(|(i, a)| {
            let p = &points[i * dim..(i + 1) * dim];
            let mut best = 0u32;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let d = sq_l2(p, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            *a = best;
        });
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            let p = &points[i * dim..(i + 1) * dim];
            for d in 0..dim {
                sums[c * dim + d] += p[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let r = (next() as usize) % n;
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&points[r * dim..(r + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }
    }
    centroids
}

/// Train a PQ codebook. `corpus` is the raw (un-normalised) flat dataset; the
/// codebook is trained on up to `TRAIN_SAMPLE` strided, unit-normalised rows.
fn train_pq(corpus: &[f32], n: usize, dim: usize, m: usize, k: usize) -> PqCodebook {
    assert_eq!(dim % m, 0, "dim must be divisible by M");
    let sub_dim = dim / m;
    let train_n = n.min(TRAIN_SAMPLE);
    let step = (n / train_n).max(1);
    // sampled + normalised training rows
    let mut train = vec![0.0f32; train_n * dim];
    for (t, row) in (0..train_n).zip((0..n).step_by(step)) {
        train[t * dim..(t + 1) * dim]
            .copy_from_slice(&normalized(&corpus[row * dim..(row + 1) * dim]));
    }
    let centroids: Vec<Vec<f32>> = (0..m)
        .into_par_iter()
        .map(|s| {
            let mut sub = vec![0.0f32; train_n * sub_dim];
            for i in 0..train_n {
                sub[i * sub_dim..(i + 1) * sub_dim]
                    .copy_from_slice(&train[i * dim + s * sub_dim..i * dim + (s + 1) * sub_dim]);
            }
            kmeans(&sub, train_n, sub_dim, k, 12, 0x9E37_79B9 ^ s as u64)
        })
        .collect();
    PqCodebook {
        m,
        k,
        sub_dim,
        centroids,
    }
}

/// PQ-quantise one raw vector (normalised inline): one centroid id per subvector.
fn quantize(cb: &PqCodebook, raw: &[f32]) -> Vec<u8> {
    let v = normalized(raw);
    let mut code = vec![0u8; cb.m];
    for s in 0..cb.m {
        let sub = &v[s * cb.sub_dim..(s + 1) * cb.sub_dim];
        let mut best = 0u8;
        let mut best_d = f32::MAX;
        for c in 0..cb.k {
            let d = sq_l2(sub, &cb.centroids[s][c * cb.sub_dim..(c + 1) * cb.sub_dim]);
            if d < best_d {
                best_d = d;
                best = c as u8;
            }
        }
        code[s] = best;
    }
    code
}

/// ADC lookup table for `query`: `lut[s * k + c]` = squared L2 of the query's
/// subvector `s` to centroid `c`. ADC distance of a code is the sum over `s`
/// of `lut[s * k + code[s]]` - it approximates squared L2 to the full vector,
/// which on unit vectors ranks identically to cosine.
fn build_lut(cb: &PqCodebook, query: &[f32]) -> Vec<f32> {
    let mut lut = vec![0.0f32; cb.m * cb.k];
    for s in 0..cb.m {
        let qsub = &query[s * cb.sub_dim..(s + 1) * cb.sub_dim];
        for c in 0..cb.k {
            lut[s * cb.k + c] = sq_l2(qsub, &cb.centroids[s][c * cb.sub_dim..(c + 1) * cb.sub_dim]);
        }
    }
    lut
}

fn adc(cb: &PqCodebook, lut: &[f32], code: &[u8]) -> f32 {
    let mut d = 0.0f32;
    for s in 0..cb.m {
        d += lut[s * cb.k + code[s] as usize];
    }
    d
}

// ── walk ────────────────────────────────────────────────────────────────────

/// Greedy beam walk over `disk`'s graph driven by `proxy(node_id) -> dist`
/// (smaller is closer). Returns the beam: up to `L_SEARCH` candidate node ids.
/// Mirrors the internal `greedy_search` (bounded beam, expand-closest-unvisited).
fn walk<P: Fn(u32) -> f32>(disk: &DiskVamanaIndex, proxy: P) -> Vec<u32> {
    let mut beam: Vec<(f32, u32)> = Vec::with_capacity(L_SEARCH + 1);
    let mut seen = std::collections::HashSet::new();
    let mut visited = std::collections::HashSet::new();

    let entry = disk.medoid();
    beam.push((proxy(entry), entry));
    seen.insert(entry);

    loop {
        let cur = beam
            .iter()
            .find(|&&(_, id)| !visited.contains(&id))
            .map(|&(_, id)| id);
        let Some(cur) = cur else { break };
        visited.insert(cur);
        for &nbr in disk.neighbors(cur) {
            if !seen.insert(nbr) {
                continue;
            }
            let d = proxy(nbr);
            let pos = beam.partition_point(|&(bd, _)| bd < d);
            beam.insert(pos, (d, nbr));
            beam.truncate(L_SEARCH);
        }
    }
    beam.into_iter().map(|(_, id)| id).collect()
}

/// Re-rank candidate node ids by exact cosine to `query`, return top-`K` ids.
fn rerank(cands: &[u32], corpus: &[f32], dim: usize, query: &[f32]) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = cands
        .iter()
        .map(|&id| {
            (
                cosine(&corpus[id as usize * dim..(id as usize + 1) * dim], query),
                id,
            )
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(K).map(|(_, id)| id).collect()
}

/// recall@K of `got` against `truth`.
fn recall(got: &[u32], truth: &[u32]) -> f64 {
    let t: std::collections::HashSet<u32> = truth.iter().copied().collect();
    got.iter().filter(|id| t.contains(id)).count() as f64 / truth.len() as f64
}

/// Build the graph, run the f32 / int8 / PQ recall@10 comparison.
fn run(label: &str, corpus: &[f32], n: usize, dim: usize, queries: &[f32], n_q: usize) {
    println!("\n=== {label}: N={n} dim={dim} query={n_q} ===");
    let t0 = std::time::Instant::now();

    let ids: Vec<u64> = (0..n as u64).collect();
    let index = VamanaIndex::build(corpus.to_vec(), ids, dim, &VamanaConfig::default());
    let tmp = tempfile::TempDir::new().expect("tempdir");
    index.save(tmp.path()).expect("save");
    println!("  build+save {:.0}s", t0.elapsed().as_secs_f64());

    // Brute-force ground truth: exact cosine top-K per query.
    let truth: Vec<Vec<u32>> = (0..n_q)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| (cosine(&corpus[i * dim..(i + 1) * dim], q), i as u32))
                .collect();
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().take(K).map(|(_, id)| id).collect()
        })
        .collect();

    // Free the in-memory index (frees its f32 vector copy + graph) before the
    // walks: at 1M that copy is ~4 GB. Every walk runs on the on-disk graph.
    drop(index);
    let disk = DiskVamanaIndex::open(tmp.path()).expect("open");

    // f32-walk: the SAME walk() as PQ, exact-cosine proxy. Per-query top-K ids
    // kept - they are the reference for the A-vs-B diagnostic (does PQ navigate
    // like exact distance, or does it return a genuinely different set?).
    let f32_ids: Vec<Vec<u32>> = (0..n_q)
        .into_par_iter()
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let cands = walk(&disk, |id| {
                1.0 - cosine(&corpus[id as usize * dim..(id as usize + 1) * dim], q)
            });
            rerank(&cands, corpus, dim, q)
        })
        .collect();
    let rec_f32 = (0..n_q)
        .map(|qi| recall(&f32_ids[qi], &truth[qi]))
        .sum::<f64>()
        / n_q as f64;

    // int8 baseline: the production on-disk int8-tier walk + f32 re-rank.
    let rec_int8 = (0..n_q)
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let got: Vec<u32> = disk
                .search(q, K)
                .expect("search")
                .iter()
                .map(|&(id, _)| id as u32)
                .collect();
            recall(&got, &truth[qi])
        })
        .sum::<f64>()
        / n_q as f64;

    println!(
        "  {:>16}{:>10}{:>14}{:>16}",
        "proxy del walk", "byte/vec", "recall@10", "overlap vs f32"
    );
    println!(
        "  {:>16}{:>10}{:>14.4}{:>16}",
        "f32-walk",
        dim * 4,
        rec_f32,
        "-"
    );
    println!(
        "  {:>16}{:>10}{:>14.4}{:>16}",
        "int8 (disk)", dim, rec_int8, "-"
    );

    // PQ: train (sampled), quantise all n, walk with ADC proxy, f32 re-rank.
    // Report recall@10 vs truth AND overlap@10 vs the f32-walk (same walk fn,
    // only the proxy differs - isolates tier navigation fidelity).
    for &m in &[64usize, 128] {
        if !dim.is_multiple_of(m) {
            continue;
        }
        let cb = train_pq(corpus, n, dim, m, 256);
        let codes: Vec<Vec<u8>> = (0..n)
            .into_par_iter()
            .map(|i| quantize(&cb, &corpus[i * dim..(i + 1) * dim]))
            .collect();
        let (sum_pq, sum_ovl): (f64, f64) = (0..n_q)
            .into_par_iter()
            .map(|qi| {
                let q = &queries[qi * dim..(qi + 1) * dim];
                let qu = normalized(q);
                let lut = build_lut(&cb, &qu);
                let cands = walk(&disk, |id| adc(&cb, &lut, &codes[id as usize]));
                let got = rerank(&cands, corpus, dim, q);
                (recall(&got, &truth[qi]), recall(&got, &f32_ids[qi]))
            })
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let rec_pq = sum_pq / n_q as f64;
        let overlap = sum_ovl / n_q as f64;
        let verdict = if rec_pq >= rec_int8 - 0.01 {
            "PASS"
        } else if rec_pq >= rec_int8 - 0.03 {
            "borderline"
        } else {
            "FAIL"
        };
        println!(
            "  {:>16}{:>10}{:>14.4}{:>16.4}  [{verdict}]",
            format!("PQ M={m} K=256"),
            m,
            rec_pq,
            overlap,
        );
    }
}

fn main() {
    eprintln!("Fase 0.c - gate recall PQ come tier di navigazione\n");
    println!("gate: PQ recall@10 entro ~1% del baseline int8 (su dati indicizzabili)");

    // mxbai reale 10K - l'ancora reale originale (testo word-salad).
    if let (Some((corpus, n, dim)), Some((queries, n_q, q_dim))) =
        (load_npy(CORPUS_NPY), load_npy(QUERY_NPY))
    {
        assert_eq!(dim, q_dim, "dim mismatch corpus/query");
        run(
            "mxbai reale 10K (testo word-salad)",
            &corpus,
            n,
            dim,
            &queries,
            n_q,
        );
    } else {
        eprintln!("  mxbai 10K npy non trovato, salto");
    }

    // Wikipedia reale 100K - l'ancora di scala su documenti reali.
    if let (Some((corpus, n, dim)), Some((queries, n_q, q_dim))) =
        (load_npy(WIKI_CORPUS_NPY), load_npy(WIKI_QUERY_NPY))
    {
        assert_eq!(dim, q_dim, "dim mismatch corpus/query");
        run(
            "Wikipedia reale 100K (documenti reali)",
            &corpus,
            n,
            dim,
            &queries,
            n_q,
        );
    } else {
        eprintln!("  Wikipedia 100K npy non trovato, salto");
    }

    // Sintetico a manifold (intrinsic dim 64): indicizzabile e isotropo,
    // proxy realistico degli embedding. Scala 100K e 1M per la degradazione.
    // Corpus e query condividono la stessa proiezione: stesso manifold.
    let dim = 1024;
    let d_in = 64;
    let proj = manifold_proj(dim, d_in, 7);
    let scales: &[usize] = if std::env::args().any(|a| a == "--with-1m") {
        &[100_000, 1_000_000]
    } else {
        &[100_000]
    };
    for &n in scales {
        let corpus = manifold_points(&proj, n, dim, d_in, 100);
        let queries = manifold_points(&proj, 200, dim, d_in, 999);
        run(
            &format!("manifold d_in={d_in} sintetico {n}"),
            &corpus,
            n,
            dim,
            &queries,
            200,
        );
    }
}
