//! skeg-bench: one unified benchmark tool so we stop rewriting ad-hoc benches
//! (and repeating the same measurement mistakes). Correct-by-construction:
//!   - recall@10 AND recall@100, both from REAL k-searches vs brute truth
//!   - RSS measured in a SUBPROCESS that opens the index but never loads the
//!     corpus (so RSS is the index's, not 2 GB of corpus + jemalloc retention)
//!   - build time, p50/p99, QPS, per (dataset, tier, config)
//!   - datasets from a registry, zero-padded to a multiple of 8
//!
//! Usage:
//!   skeg-bench --dataset mxbai500k --tier tq2 --tier tq1 [--n N] [--nq NQ]
//!              [--l-search 300] [--rerank 800] [--mmap]
//!   (internal) skeg-bench --measure DIR BITS MMAP QPATH DIM NQ LSEARCH RERANK

use ahash::AHashSet;
use rayon::prelude::*;
use skeg_simd::cosine_f32;
use skeg_vector::{DiskVamanaIndex, QuantKind};
use std::path::{Path, PathBuf};

/// Workspace root (skeg-bench lives under crates/skeg-vector).
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn resolve_dataset_path(workspace: &Path, bench_data_dir: Option<&Path>, path: &str) -> PathBuf {
    const BENCH_DATA_PREFIX: &str = "skeg-bench-data/";
    if let Some(relative) = path.strip_prefix(BENCH_DATA_PREFIX) {
        return match bench_data_dir {
            Some(directory) => directory.join(relative),
            None => workspace.join(BENCH_DATA_PREFIX).join(relative),
        };
    }
    workspace.join(path)
}

/// Resolve a registered dataset path. Set `SKEG_BENCH_DATA_DIR` to the
/// directory holding the compact f16 datasets; all existing `.npy` paths
/// remain relative to the workspace.
fn dataset_path(path: &str) -> PathBuf {
    let workspace = root();
    let bench_data_dir = std::env::var_os("SKEG_BENCH_DATA_DIR").map(PathBuf::from);
    resolve_dataset_path(&workspace, bench_data_dir.as_deref(), path)
}

/// (name, corpus rel-path, query rel-path, native dim).
const DATASETS: &[(&str, &str, &str, usize)] = &[
    (
        "glove",
        "skeg-bench/data/glove_corpus.npy",
        "skeg-bench/data/glove_queries.npy",
        100,
    ),
    (
        "minilm",
        "skeg/bench-compare/embeddings_cache/corpus_minilm-wiki.npy",
        "skeg/bench-compare/embeddings_cache/queries_minilm-wiki_200.npy",
        384,
    ),
    (
        "mnist",
        "skeg-bench/data/mnist_corpus_60k.npy",
        "skeg-bench/data/mnist_queries_200.npy",
        784,
    ),
    (
        "mxbai",
        "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki.npy",
        "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki_200.npy",
        1024,
    ),
    (
        "mxbai500k",
        "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_500k.npy",
        "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy",
        1024,
    ),
    (
        "mxbai1m",
        "skeg/bench-compare/embeddings_cache/corpus_mxbai-wiki-chunked_1m.npy",
        "skeg/bench-compare/embeddings_cache/queries_mxbai-wiki-chunked_1000.npy",
        1024,
    ),
    (
        "qwen",
        "skeg/bench-compare/embeddings_cache/corpus_qwen3emb4b_100k.npy",
        "skeg/bench-compare/embeddings_cache/queries_qwen3emb4b_1k.npy",
        2560,
    ),
    // Real embeddings uploaded for the x86 validation run (not synthetic -
    // arxiv-instructorxl/gemini-001/openai3-large/wiki-cohere real corpora,
    // converted to a raw f16 format to shrink the transfer; loaded via
    // `load_f16` below, detected by the `.f16` extension).
    (
        "arxiv-instructorxl",
        "skeg-bench-data/arxiv-instructorxl_corpus.f16",
        "skeg-bench-data/arxiv-instructorxl_queries.f16",
        768,
    ),
    (
        "gemini-001",
        "skeg-bench-data/gemini-001_corpus.f16",
        "skeg-bench-data/gemini-001_queries.f16",
        768,
    ),
    (
        "openai3-large",
        "skeg-bench-data/openai3-large_corpus.f16",
        "skeg-bench-data/openai3-large_queries.f16",
        1536,
    ),
    (
        "wiki-cohere-shuf",
        "skeg-bench-data/wiki-cohere-shuf_corpus.f16",
        "skeg-bench-data/wiki-cohere-shuf_queries.f16",
        1024,
    ),
    (
        "glove104-500k",
        "skeg-bench-data/glove104_corpus.f16",
        "skeg-bench-data/glove104_queries.f16",
        104,
    ),
    (
        "mxbai-500k",
        "skeg-bench-data/mxbai-500k_corpus.f16",
        "skeg-bench-data/mxbai-500k_queries.f16",
        1024,
    ),
];

/// Decode one IEEE 754 binary16 value to f32 (hand-rolled: avoids adding the
/// `half` crate just for this one-off real-embeddings transfer format).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x3FF);
    let abs = if exp == 0 {
        (frac as f32) * 2f32.powi(-24) // subnormal (or zero when frac == 0)
    } else if exp == 0x1F {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -abs } else { abs }
}

/// Load the raw f16 format written by the local conversion script:
/// `[u32 n][u32 dim][u32 reserved]` then `n * dim` little-endian f16 values.
fn decode_f16_dataset(
    bytes: &[u8],
    cap: usize,
    pad: usize,
) -> Result<(Vec<Vec<f32>>, usize), String> {
    const HEADER_LEN: usize = 12;
    if bytes.len() < HEADER_LEN {
        return Err("f16 dataset header is shorter than 12 bytes".into());
    }
    let rows = u32::from_le_bytes(bytes[0..4].try_into().expect("header is checked")) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().expect("header is checked")) as usize;
    if pad < dim {
        return Err(format!(
            "f16 dataset dimension {dim} exceeds padded dimension {pad}"
        ));
    }
    let payload_len = rows
        .checked_mul(dim)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| "f16 dataset dimensions overflow usize".to_owned())?;
    let required_len = HEADER_LEN
        .checked_add(payload_len)
        .ok_or_else(|| "f16 dataset length overflows usize".to_owned())?;
    if bytes.len() < required_len {
        return Err(format!(
            "f16 dataset payload is truncated: expected {required_len} bytes, found {}",
            bytes.len()
        ));
    }

    let n = cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; pad];
            for (j, value) in v.iter_mut().take(dim).enumerate() {
                let off = HEADER_LEN + (i * dim + j) * 2;
                let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                *value = f16_to_f32(bits);
            }
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    Ok((out, dim))
}

fn load_f16(path: &Path, cap: usize, pad: usize) -> (Vec<Vec<f32>>, usize) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("cannot read f16 dataset {}: {error}", path.display()));
    decode_f16_dataset(&bytes, cap, pad)
        .unwrap_or_else(|error| panic!("cannot decode f16 dataset {}: {error}", path.display()))
}

/// Dispatch to the right loader by extension - `.f16` (raw, this run's real
/// embeddings) or `.npy` (everything already in the registry above).
fn load_dataset(path: &Path, cap: usize, pad: usize) -> (Vec<Vec<f32>>, usize) {
    if path.extension().and_then(|e| e.to_str()) == Some("f16") {
        load_f16(path, cap, pad)
    } else {
        load_npy(path, cap, pad)
    }
}

fn tier_of(s: &str) -> (QuantKind, u8) {
    match s {
        "tq1" => (QuantKind::TurboQuant { bits: 1 }, 1),
        "tq2" => (QuantKind::TurboQuant { bits: 2 }, 2),
        "tq4" => (QuantKind::TurboQuant { bits: 4 }, 4),
        _ => panic!("unknown tier '{s}' (tq1|tq2|tq4)"),
    }
}

fn load_npy(path: &Path, cap: usize, pad: usize) -> (Vec<Vec<f32>>, usize) {
    let bytes =
        std::fs::read(path).unwrap_or_else(|_| panic!("missing dataset: {}", path.display()));
    let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + hl]).unwrap();
    let sh = header.find("'shape':").unwrap();
    let lp = header[sh..].find('(').unwrap() + sh + 1;
    let rp = header[lp..].find(')').unwrap() + lp;
    let dims: Vec<usize> = header[lp..rp]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (rows, dim) = (dims[0], dims[1]);
    let data: Vec<f32> = bytes[10 + hl..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let n = cap.min(rows);
    let out = (0..n)
        .map(|i| {
            let mut v = vec![0.0f32; pad];
            v[..dim].copy_from_slice(&data[i * dim..i * dim + dim]);
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.iter_mut().for_each(|x| *x /= nrm);
            v
        })
        .collect();
    (out, dim)
}

fn truth(corpus: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<AHashSet<u64>> {
    queries
        .par_iter()
        .map(|q| {
            let mut t: Vec<(f32, u64)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (cosine_f32(q, v), i as u64))
                .collect();
            t.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            t.iter().take(k).map(|&(_, id)| id).collect()
        })
        .collect()
}

fn rss_mib() -> f64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

/// Subprocess entry: open the pre-built index (NO corpus) and report RSS + p50 +
/// p99 + qps for a k=10 search. Keeps RSS free of corpus/jemalloc pollution.
fn measure_mode(args: &[String]) {
    let dir = PathBuf::from(&args[0]);
    let bits: u8 = args[1].parse().unwrap();
    let mmap = args[2] == "1";
    let qpath = PathBuf::from(&args[3]);
    let dim: usize = args[4].parse().unwrap();
    let nq: usize = args[5].parse().unwrap();
    let ls: usize = args[6].parse().unwrap();
    let rr: usize = args[7].parse().unwrap();
    let (queries, _) = load_dataset(&qpath, nq, dim);
    let idx =
        DiskVamanaIndex::open_with_tier_full(&dir, QuantKind::TurboQuant { bits }, mmap, false)
            .unwrap();
    for q in queries.iter().take(32) {
        idx.search_with_params(q, 10, ls, rr).unwrap();
    }
    let mut lat: Vec<f64> = Vec::with_capacity(queries.len());
    for q in &queries {
        let s = std::time::Instant::now();
        idx.search_with_params(q, 10, ls, rr).unwrap();
        lat.push(s.elapsed().as_secs_f64() * 1e6);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let p50 = lat[lat.len() / 2];
    let p99 = lat[(lat.len() as f64 * 0.99) as usize];
    let qps = 1e6 / (lat.iter().sum::<f64>() / lat.len() as f64);
    let cqps = concurrent_qps(&idx, &queries, ls, rr, num_cpus());
    // Parseable line for the parent.
    println!(
        "MEASURE rss={:.0} p50={p50:.0} p99={p99:.0} qps={qps:.0} cqps={cqps:.0}",
        rss_mib()
    );
}

/// Real concurrent throughput: `threads` OS threads hammer the shared,
/// already-built index with the query set for ~2s each, summing completed
/// queries over the actual wall time. Unlike `qps` (1 / mean single-thread
/// latency), this is a genuine measurement of how many searches/sec the
/// machine sustains under load - the number that matters for a live server.
fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}

fn concurrent_qps(
    idx: &DiskVamanaIndex,
    queries: &[Vec<f32>],
    l_search: usize,
    rerank: usize,
    threads: usize,
) -> f64 {
    let (done, elapsed) = run_concurrent(
        threads,
        std::time::Duration::from_secs(2),
        |worker, iteration| {
            let query = &queries[(worker + iteration as usize * threads) % queries.len()];
            idx.search_with_params(query, 10, l_search, rerank).unwrap();
        },
    );
    done as f64 / elapsed.as_secs_f64()
}

/// Execute benchmark work concurrently for a bounded period. The two barriers
/// ensure the timed interval starts only after every worker is ready.
fn run_concurrent<F>(
    threads: usize,
    duration: std::time::Duration,
    work: F,
) -> (u64, std::time::Duration)
where
    F: Fn(usize, u64) + Sync,
{
    let threads = threads.max(1);
    let done = std::sync::atomic::AtomicU64::new(0);
    let ready = std::sync::Barrier::new(threads + 1);
    let start_gate = std::sync::Barrier::new(threads + 1);
    let started = std::sync::Mutex::new(None::<std::time::Instant>);
    std::thread::scope(|s| {
        for worker in 0..threads {
            let done = &done;
            let work = &work;
            let ready = &ready;
            let start_gate = &start_gate;
            let started = &started;
            s.spawn(move || {
                ready.wait();
                start_gate.wait();
                let start = started
                    .lock()
                    .unwrap()
                    .expect("start is set before release");
                for iteration in 0.. {
                    work(worker, iteration);
                    done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if start.elapsed() >= duration {
                        break;
                    }
                }
            });
        }
        ready.wait();
        let start = std::time::Instant::now();
        *started.lock().unwrap() = Some(start);
        start_gate.wait();
    });
    let start = started
        .lock()
        .unwrap()
        .expect("start is set after worker scope");
    let elapsed = start.elapsed();
    (done.load(std::sync::atomic::Ordering::Relaxed), elapsed)
}

struct Args {
    datasets: Vec<String>,
    tiers: Vec<String>,
    n: usize,
    nq: usize,
    l_search: usize,
    rerank: usize,
    mmap: bool,
}

fn parse_args(argv: &[String]) -> Args {
    let mut a = Args {
        datasets: vec![],
        tiers: vec![],
        n: usize::MAX,
        nq: 200,
        l_search: 300,
        rerank: 800,
        mmap: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let next = || argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--dataset" => a.datasets.push(next()),
            "--tier" => a.tiers.push(next()),
            "--n" => a.n = next().parse().unwrap(),
            "--nq" => a.nq = next().parse().unwrap(),
            "--l-search" => a.l_search = next().parse().unwrap(),
            "--rerank" => a.rerank = next().parse().unwrap(),
            "--mmap" => {
                a.mmap = true;
                i += 1;
                continue;
            }
            other => panic!("unknown arg '{other}'"),
        }
        i += 2;
    }
    if a.datasets.is_empty() {
        a.datasets.push("mxbai".into());
    }
    if a.tiers.is_empty() {
        a.tiers = vec!["tq1".into(), "tq2".into()];
    }
    a
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("--measure") {
        measure_mode(&argv[1..]);
        return;
    }
    let a = parse_args(&argv);
    let exe = std::env::current_exe().unwrap();
    println!(
        "RSS in a subprocess (no corpus). recall = real k-searches. mmap={}",
        a.mmap
    );
    println!(
        "{:<11} {:<4} {:>4} {:>8} {:>7} {:>6} {:>9} {:>10} {:>7} {:>7} {:>5} {:>6}",
        "dataset",
        "tier",
        "dim",
        "n",
        "build_s",
        "RSS",
        "recall@10",
        "recall@100",
        "p50us",
        "p99us",
        "qps",
        "cqps"
    );
    for dname in &a.datasets {
        let &(_, cpath, qpath, native) = DATASETS
            .iter()
            .find(|d| d.0 == dname)
            .unwrap_or_else(|| panic!("unknown dataset '{dname}'"));
        let pad = native.next_multiple_of(8);
        let corpus_path = dataset_path(cpath);
        let query_path = dataset_path(qpath);
        let (corpus, dim) = load_dataset(&corpus_path, a.n, pad);
        let (queries, _) = load_dataset(&query_path, a.nq, pad);
        let n = corpus.len();
        let t10 = truth(&corpus, &queries, 10);
        let t100 = truth(&corpus, &queries, 100);
        for tname in &a.tiers {
            let (tier, bits) = tier_of(tname);
            let dir = std::env::temp_dir().join(format!("skeg_bench_{dname}_{bits}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let t = std::time::Instant::now();
            let mut idx = DiskVamanaIndex::create_empty_with_tier(&dir, dim, 300, tier).unwrap();
            for (id, v) in corpus.iter().enumerate() {
                idx.insert(id as u64, v).unwrap();
            }
            idx.consolidate().unwrap();
            let build_s = t.elapsed().as_secs_f64();
            let r = |k: usize, tr: &[AHashSet<u64>], rr: usize| -> f64 {
                let mut h = 0usize;
                for (q, t) in queries.iter().zip(tr) {
                    h += idx
                        .search_with_params(q, k, a.l_search, rr)
                        .unwrap()
                        .iter()
                        .filter(|(id, _)| t.contains(id))
                        .count();
                }
                h as f64 / (queries.len() * k) as f64
            };
            let r10 = r(10, &t10, a.rerank.max(80));
            let r100 = r(100, &t100, a.rerank);
            drop(idx);
            // RSS + latency in a clean subprocess (no corpus in RAM).
            let out = std::process::Command::new(&exe)
                .args([
                    "--measure",
                    dir.to_str().unwrap(),
                    &bits.to_string(),
                    if a.mmap { "1" } else { "0" },
                    &query_path.to_string_lossy(),
                    &pad.to_string(),
                    &a.nq.to_string(),
                    &a.l_search.to_string(),
                    &a.rerank.max(80).to_string(),
                ])
                .output()
                .unwrap();
            let line = String::from_utf8_lossy(&out.stdout);
            let get = |k: &str| {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(k))
                    .unwrap_or("?")
                    .to_string()
            };
            println!(
                "{dname:<11} {tname:<4} {pad:>4} {n:>8} {build_s:>7.0} {:>6} {r10:>9.4} {r100:>10.4} {:>7} {:>7} {:>5} {:>6}",
                get("rss="),
                get("p50="),
                get("p99="),
                get("qps="),
                get("cqps=")
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_f16_dataset, resolve_dataset_path, run_concurrent};
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn decode_f16_dataset_normalizes_and_pads_rows() {
        let bytes = [
            1, 0, 0, 0, // rows
            2, 0, 0, 0, // dim
            0, 0, 0, 0, // reserved
            0, 0x3c, // 1.0
            0, 0x40, // 2.0
        ];

        let (rows, dim) = decode_f16_dataset(&bytes, usize::MAX, 8).unwrap();

        assert_eq!(dim, 2);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 8);
        assert!((rows[0][0] - 1.0 / 5.0_f32.sqrt()).abs() < 1e-6);
        assert!((rows[0][1] - 2.0 / 5.0_f32.sqrt()).abs() < 1e-6);
        assert_eq!(&rows[0][2..], &[0.0; 6]);
    }

    #[test]
    fn decode_f16_dataset_rejects_truncated_header() {
        let err = decode_f16_dataset(&[1, 0, 0], 1, 8).unwrap_err();

        assert!(err.contains("header"));
    }

    #[test]
    fn resolve_dataset_path_uses_explicit_p4_data_directory() {
        let path = resolve_dataset_path(
            Path::new("/workspace/skeg"),
            Some(Path::new("/mnt/bench-data")),
            "skeg-bench-data/mxbai-500k_corpus.f16",
        );

        assert_eq!(path, Path::new("/mnt/bench-data/mxbai-500k_corpus.f16"));
    }

    #[test]
    fn run_concurrent_starts_every_requested_worker() {
        let seen = AtomicUsize::new(0);

        run_concurrent(4, Duration::from_millis(10), |worker, _| {
            seen.fetch_or(1 << worker, Ordering::Relaxed);
        });

        assert_eq!(seen.load(Ordering::Relaxed), 0b1111);
    }
}
