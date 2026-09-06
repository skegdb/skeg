//! Bench `Tenant::query_filtered` across record counts. DiskVamana
//! backend; oversample + post-filter is the adapter's v0.1 path.
//!
//! Run with:
//!   cargo bench --bench tenant_query

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use skeg_rigging::{Filter, QueryFiltered, RecordId, RecordMeta, TenantId};
use skeg_rigging_skeg::Tenant;

const DIM: usize = 32;

fn synth_vector(seed: u64) -> Vec<f32> {
    (0..DIM)
        .map(|d| {
            let h =
                ((seed as u32).wrapping_mul(2654435761) ^ (d as u32).wrapping_mul(40503)) as f32;
            (h.sin() + 1.0) * 0.5
        })
        .collect()
}

struct AcceptShareable;
impl Filter for AcceptShareable {
    fn accept(&self, m: &RecordMeta<'_>) -> bool {
        m.shareable
    }
}

fn populate(tmp: &tempfile::TempDir, n: u64) -> Tenant {
    let tenant = Tenant::create_new(tmp.path(), TenantId::ZERO, DIM as u32).expect("create");
    for i in 0..n {
        tenant
            .insert(
                RecordId(i),
                synth_vector(i),
                i % 2 == 0,
                vec![],
                format!("payload {i}").into_bytes(),
            )
            .expect("insert");
    }
    tenant.flush().expect("flush");
    tenant
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_query");
    group.sample_size(20);
    for &n in &[100u64, 1_000, 5_000] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let tenant = populate(&tmp, n);
        let query = synth_vector(0); // close to record 0
        group.bench_with_input(
            BenchmarkId::new("records", n),
            &(tenant, query),
            |b, (t, q)| {
                b.iter(|| {
                    let hits = t
                        .query_filtered(black_box(q), 10, &AcceptShareable)
                        .expect("query");
                    black_box(hits);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
