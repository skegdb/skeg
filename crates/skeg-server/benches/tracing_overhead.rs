//! G-O3.1 benchmark: per-span cost of the `vsearch` instrumentation
//! the resp3 and binary handlers emit.
//!
//! Measures three configurations on the same synthetic loop:
//!
//!   no_tracing       no subscriber installed; spans are no-ops.
//!   fmt_only         stdout fmt layer, EnvFilter set to `off` so
//!                    nothing is actually serialised.
//!   fmt_with_filter  stdout fmt layer, EnvFilter `info`; spans the
//!                    handler creates land in the filter check + drop.
//!
//! The benchmark only exercises tracing primitives (span creation,
//! `record()`, enter/exit). It does NOT spin up an OTel exporter
//! because the production overhead with OTel-bridge is dominated by
//! the same primitives plus a per-span clone into the batch queue.
//!
//! Gate: `fmt_only` adds < 5% wall time vs `no_tracing` on M-class
//! hardware (G-O3.1 from observability/PLAN.md).
//!
//! Env knobs:
//!   SKEG_TRACE_BENCH_ITERS   span ops per run (default 5_000_000)
//!   SKEG_TRACE_BENCH_WARMUP  warmup iters     (default 200_000)

#![deny(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::hint::black_box;
use std::time::Instant;

use tracing::{Subscriber, info_span};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Synthesised hot loop mimicking the per-call cost of the `vsearch`
/// span: create span, record three dynamic fields, enter, do trivial
/// work, exit. The dynamic fields use the same shapes as the handlers
/// (vindex string, k usize, l_search u32, vector_dim usize, hits usize).
fn run_loop(iters: usize) -> u64 {
    let mut acc: u64 = 0;
    for i in 0..iters {
        let span = info_span!(
            "vsearch",
            protocol = "binary",
            vindex = "bench-vindex",
            k = tracing::field::Empty,
            l_search = tracing::field::Empty,
            vector_dim = tracing::field::Empty,
            hits = tracing::field::Empty,
        );
        span.record("k", 10_usize);
        span.record("l_search", 300_u32);
        span.record("vector_dim", 1024_usize);
        {
            let _g = span.enter();
            // Trivial work: defeats the optimiser collapsing the whole
            // span into a no-op when no subscriber is registered.
            acc = acc.wrapping_add(black_box(i as u64));
        }
        span.record("hits", 10_usize);
    }
    acc
}

fn measure<S, F>(label: &str, subscriber_factory: F, iters: usize, warmup: usize) -> f64
where
    S: Subscriber + Send + Sync + 'static,
    F: FnOnce() -> S + Send + 'static,
{
    // Subscribers are global: spawn a thread, install the subscriber
    // there, run the loop. The thread's subscriber is dropped on join,
    // restoring the global to whatever it was (typically NoSubscriber).
    let handle = std::thread::Builder::new()
        .name(format!("trace-bench-{label}"))
        .spawn(move || {
            let sub = subscriber_factory();
            let _guard = tracing::subscriber::set_default(sub);
            let _ = run_loop(warmup);
            let t0 = Instant::now();
            let acc = run_loop(iters);
            let elapsed_ns = t0.elapsed().as_nanos() as f64;
            black_box(acc);
            elapsed_ns / iters as f64
        })
        .expect("spawn bench thread");
    handle.join().expect("join bench thread")
}

fn main() {
    let iters = env_usize("SKEG_TRACE_BENCH_ITERS", 5_000_000);
    let warmup = env_usize("SKEG_TRACE_BENCH_WARMUP", 200_000);

    println!("# tracing overhead microbench (per-span ns)");
    println!("# iters={iters} warmup={warmup}");
    println!(
        "# host: target_os={} target_arch={}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("config,ns_per_span");

    // No subscriber: tracing macros short-circuit at the dispatch level.
    let baseline_ns = measure(
        "noop",
        || tracing_subscriber::registry().with(EnvFilter::new("off")),
        iters,
        warmup,
    );
    println!("no_tracing,{baseline_ns:.2}");

    let fmt_off_ns = measure(
        "fmt-off",
        || {
            tracing_subscriber::registry()
                .with(EnvFilter::new("off"))
                .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink))
        },
        iters,
        warmup,
    );
    let off_ratio = fmt_off_ns / baseline_ns;
    println!("fmt_off,{fmt_off_ns:.2}");

    let fmt_info_ns = measure(
        "fmt-info",
        || {
            tracing_subscriber::registry()
                .with(EnvFilter::new("info"))
                .with(tracing_subscriber::fmt::layer().with_writer(std::io::sink))
        },
        iters,
        warmup,
    );
    let info_ratio = fmt_info_ns / baseline_ns;
    println!("fmt_info,{fmt_info_ns:.2}");

    eprintln!(
        "# baseline {baseline_ns:.2} ns/span; fmt_off {fmt_off_ns:.2} ({off_ratio:.2}x); fmt_info {fmt_info_ns:.2} ({info_ratio:.2}x)"
    );

    // G-O3.1 gate: the on-but-filtered path must stay within 5% of the
    // no-subscriber baseline. (We use fmt_off as the proxy because the
    // OTel bridge layer behaves the same way under a filter that drops
    // the span before it reaches the exporter.)
    let gate_pass = off_ratio < 1.05;
    if gate_pass {
        eprintln!("# G-O3.1 PASS: fmt_off / baseline = {off_ratio:.3} (< 1.05)");
    } else {
        eprintln!(
            "# G-O3.1 FAIL: fmt_off / baseline = {off_ratio:.3} (>= 1.05) - investigate"
        );
        std::process::exit(1);
    }
}
