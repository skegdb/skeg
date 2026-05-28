//! In-process text dumper for the RESP3 `STATS` command.
//!
//! Produces a Prometheus-flavoured text format on demand. The output is
//! deterministic given a snapshot of the static counters: an entire dump
//! is taken with sequential `load(Relaxed)` reads, so different lines
//! may reflect counters captured a few nanoseconds apart, but never a
//! torn 64-bit value.
//!
//! Called by the server's `STATS` handler and by the `http` exporter
//! when `/metrics` is hit. Both paths go through this function so the
//! schema stays consistent.

use core::fmt::Write;

use crate::{
    Counter, Gauge, Op,
    histograms::{self, BUCKET_BOUNDS_US, BUCKETS},
    metrics,
};

/// Serialise the full metric set into a `String` in Prometheus text
/// format.
///
/// The result is suitable as a `STATS` reply body and as the response
/// to `GET /metrics`. Lines are LF-terminated and end with a final LF.
pub fn dump_text() -> String {
    let mut out = String::with_capacity(4096);

    // Per-op counters (sum across shards, plus per-shard if you want).
    out.push_str("# HELP skeg_ops_total Total number of operations served.\n");
    out.push_str("# TYPE skeg_ops_total counter\n");
    for &op in &Op::ALL {
        let total = metrics::op_total(op);
        let _ = writeln!(&mut out, "skeg_ops_total{{op=\"{}\"}} {}", op.name(), total);
    }

    // Per-op duration histograms (Prometheus histogram convention:
    // cumulative bucket counts + `_count` + `_sum`).
    out.push_str("\n# HELP skeg_op_duration_seconds Operation duration in seconds.\n");
    out.push_str("# TYPE skeg_op_duration_seconds histogram\n");
    for &op in &Op::ALL {
        let mut cumulative: u64 = 0;
        for b in 0..BUCKETS {
            cumulative += histograms::bucket(op, b);
            let le_us = BUCKET_BOUNDS_US[b];
            // Prometheus convention: last bucket is `+Inf`.
            let le_str = if b == BUCKETS - 1 {
                String::from("+Inf")
            } else {
                // Render in seconds (le_us is the exclusive upper edge,
                // we report seconds with enough precision to be unique).
                let secs = le_us as f64 / 1_000_000.0;
                format!("{:.6}", secs)
            };
            let _ = writeln!(
                &mut out,
                "skeg_op_duration_seconds_bucket{{op=\"{}\",le=\"{}\"}} {}",
                op.name(),
                le_str,
                cumulative
            );
        }
        let count = histograms::count(op);
        let sum_secs = histograms::sum_us(op) as f64 / 1_000_000.0;
        let _ = writeln!(
            &mut out,
            "skeg_op_duration_seconds_count{{op=\"{}\"}} {}",
            op.name(),
            count
        );
        let _ = writeln!(
            &mut out,
            "skeg_op_duration_seconds_sum{{op=\"{}\"}} {}",
            op.name(),
            sum_secs
        );
    }

    // Global counters.
    out.push('\n');
    for &c in &Counter::ALL {
        let _ = writeln!(&mut out, "# TYPE {} counter", c.name());
        let _ = writeln!(&mut out, "{} {}", c.name(), metrics::counter(c));
    }

    // Gauges.
    out.push('\n');
    for &g in &Gauge::ALL {
        let _ = writeln!(&mut out, "# TYPE {} gauge", g.name());
        let _ = writeln!(&mut out, "{} {}", g.name(), metrics::gauge(g));
    }

    out
}
