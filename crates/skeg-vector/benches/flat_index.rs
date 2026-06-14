//! Flat-scan vector search benchmarks.
//!
//! Stressful by design: each case scans a large vector set per search, the
//! inner loop of a flat VSEARCH. Throughput is reported in vectors scanned
//! per second (Melem/s).

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use skeg_vector::{FlatIndex, QuantKind};

const DIM: usize = 1536; // typical embedding dimension

fn random_index(n: usize, kind: QuantKind, seed: u64) -> FlatIndex {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx = FlatIndex::new(DIM, kind);
    let mut v = vec![0.0f32; DIM];
    for id in 0..n {
        for x in &mut v {
            *x = rng.random_range(-1.0..1.0);
        }
        idx.insert(id as u64, &v);
    }
    idx
}

fn random_query(seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..DIM).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn bench_search(c: &mut Criterion) {
    let cases = [
        ("f32_50k", 50_000usize, QuantKind::F32),
        ("int8_100k", 100_000, QuantKind::Int8),
        ("binary_200k", 200_000, QuantKind::Binary),
    ];
    let mut g = c.benchmark_group("flat_search");
    g.sample_size(20);
    for (name, n, kind) in cases {
        let mut idx = random_index(n, kind, 1);
        let query = random_query(2);
        // Warm up: the first search builds the quantized form (int8/binary).
        let _ = idx.search(&query, 10);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_function(name, |b| {
            b.iter(|| black_box(idx.search(black_box(&query), 10)));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
