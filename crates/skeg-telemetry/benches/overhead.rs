//! Hot-path overhead gate.
//!
//! Each `record_op` call should complete in <50 ns on Apple Silicon.
//! Criterion fails the build if any of the gated benches drift past
//! that ceiling, so adding a metric never silently regresses query
//! latency.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use skeg_telemetry::{record_op, Op};

fn bench_record_op(c: &mut Criterion) {
    let dur = Duration::from_micros(123);
    c.bench_function("record_op", |b| {
        b.iter(|| {
            record_op(black_box(Op::VSearch), black_box(0), black_box(dur));
        });
    });
}

fn bench_record_op_4_shards(c: &mut Criterion) {
    let dur = Duration::from_micros(200);
    c.bench_function("record_op_4_shards", |b| {
        b.iter(|| {
            for shard in 0u16..4 {
                record_op(black_box(Op::VSearch), black_box(shard), black_box(dur));
            }
        });
    });
}

fn bench_record_op_various_durations(c: &mut Criterion) {
    c.bench_function("record_op_various_durations", |b| {
        let durs = [
            Duration::from_nanos(500),
            Duration::from_micros(1),
            Duration::from_micros(64),
            Duration::from_millis(1),
            Duration::from_millis(64),
            Duration::from_secs(1),
        ];
        b.iter(|| {
            for &d in &durs {
                record_op(black_box(Op::VSearch), black_box(0), black_box(d));
            }
        });
    });
}

criterion_group!(
    name = overhead;
    config = Criterion::default()
        .sample_size(200)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_record_op, bench_record_op_4_shards, bench_record_op_various_durations
);
criterion_main!(overhead);
