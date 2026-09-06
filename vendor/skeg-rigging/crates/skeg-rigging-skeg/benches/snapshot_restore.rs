//! Bench `TenantLifecycle::snapshot` + `Tenant::restore_from`
//! across record counts. This measures what an orchestrator pays
//! for a tenant backup-and-restore cycle.
//!
//! Run with:
//!   cargo bench --bench snapshot_restore

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use skeg_rigging::{IterVectors, RecordId, TenantId, TenantLifecycle};
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

fn populate_into(path: &std::path::Path, n: u64) -> Tenant {
    let t = Tenant::create_new(path, TenantId::ZERO, DIM as u32).expect("create");
    for i in 0..n {
        t.insert(
            RecordId(i),
            synth_vector(i),
            true,
            vec![],
            format!("p{i}").into_bytes(),
        )
        .expect("insert");
    }
    t.flush().expect("flush");
    t
}

fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot");
    group.sample_size(20);
    for &n in &[100u64, 1_000, 5_000] {
        group.bench_with_input(BenchmarkId::new("records", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let src = tempfile::tempdir().expect("src");
                    let snap = tempfile::tempdir().expect("snap");
                    let t = populate_into(src.path(), n);
                    (src, snap, t)
                },
                |(src, snap, t)| {
                    let boxed: Box<dyn TenantLifecycle> = Box::new(t);
                    boxed.snapshot(snap.path()).expect("snapshot");
                    black_box((src, snap));
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn bench_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("restore");
    group.sample_size(20);
    for &n in &[100u64, 1_000, 5_000] {
        group.bench_with_input(BenchmarkId::new("records", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let src = tempfile::tempdir().expect("src");
                    let snap = tempfile::tempdir().expect("snap");
                    let dest_dir = tempfile::tempdir().expect("dest");
                    let t = populate_into(src.path(), n);
                    let boxed: Box<dyn TenantLifecycle> = Box::new(t);
                    boxed.snapshot(snap.path()).expect("snapshot");
                    (src, snap, dest_dir)
                },
                |(src, snap, dest_dir)| {
                    let restored =
                        Tenant::restore_from(snap.path(), dest_dir.path()).expect("restore");
                    let count = <Tenant as IterVectors>::record_count(&restored);
                    black_box((src, snap, dest_dir, count));
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_snapshot, bench_restore);
criterion_main!(benches);
