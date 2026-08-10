//! WAL append baseline. Run `cargo bench -p skeg-vector --bench wal_append`.

use criterion::{Criterion, criterion_group, criterion_main};
use skeg_vector::DiskVamanaIndex;
use tempfile::TempDir;

const DIM: usize = 1024;

fn bench_wal_append(c: &mut Criterion) {
    let dir = TempDir::new().expect("temporary index directory");
    let mut index = DiskVamanaIndex::create_empty(dir.path(), DIM, 64).expect("create disk index");
    index.set_auto_flush(false);
    let vector = vec![0.125f32; DIM];

    c.bench_function("wal_append/1024d", |b| {
        b.iter(|| index.insert(0, &vector).expect("append vector WAL record"));
    });
}

criterion_group!(benches, bench_wal_append);
criterion_main!(benches);
