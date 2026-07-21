#![deny(unsafe_code)]

//! skeg-server-resp3: same engine as the `skeg` binary (binary protocol),
//! but speaks Redis wire (RESP2/RESP3) on the listener.
//!
//! Differences from the `skeg` binary:
//! - Default address `127.0.0.1:6379` (Redis port), not `7379`.
//! - `Server::run_resp3()` instead of `run()`.
//!
//! This binary ships single-tenant. The multi-tenant flavour lives in a
//! separate crate (`skeg-server-tenant`), which installs a `TenantBackend`
//! on top of this engine.

use skeg_server::Server;
use skeg_vector::QuantKind;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[allow(non_upper_case_globals, unsafe_code)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: Option<&'static core::ffi::c_char> = Some(
    // SAFETY: identical to the main `skeg` binary - C string literal,
    // NUL-terminated, jemalloc reads as `const char *`.
    unsafe { &*c"background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0".as_ptr() },
);

const HELP: &str = concat!(
    "\
skeg-resp3 ",
    env!("CARGO_PKG_VERSION"),
    "

USAGE:
    skeg-resp3 [OPTIONS]

OPTIONS:
    --addr <HOST:PORT>     Listen address. Default 127.0.0.1:6379 (Redis port).
                             Env: SKEG_RESP3_ADDR.
    --data-dir <PATH>      Data directory. Default ./data. Env: SKEG_DATA_DIR.
    --mode <MODE>          'rw' (default) or 'serve' (read-only, mmap tier).
    --tier <KIND>          Quantizer for the serve tier:
                             tq2 (default) | int8 | pq | pq:M:K |
                             turboquant-1 | turboquant-2 | turboquant-4
                             (aliases: tq1, tq2, tq4). Same default applies to
                             SKEG.VINDEX.CREATE when its kind arg is omitted.
    --speed                Opt-in early-termination in greedy walk
                             (-0.3 to -0.7% recall@10, +40-60% QPS).
                             Also: SKEG_SPEED=1.
    --workers <N>          Dispatch SKEG.VSEARCH to a worker pool (N threads).
                             0 (default) = inline on shard. Also: SKEG_WORKERS.
    --tier-mmap            mmap the TurboQuant tier. Also: SKEG_TIER_MMAP=1.
    --graph-mmap           mmap the Vamana graph Node array. Also: SKEG_GRAPH_MMAP=1.
    -h, --help             Print this help.
    -V, --version          Print the version.

PROTOCOL: RESP3 (Redis-compatible). Use redis-cli, redis-py, etc.
DOCS:     https://github.com/skegdb/skeg
"
);

/// Tracing init: fmt layer to stdout, plus an optional OTLP/gRPC layer
/// when the `tracing-otlp` feature is on AND
/// `SKEG_TRACE_OTLP_ENDPOINT` is set. Mirrors the wiring in the `skeg`
/// binary so both listeners produce equivalent spans.
fn init_tracing() -> Result<(), Box<dyn std::error::Error>> {
    let env_filter = EnvFilter::from_default_env();
    let fmt_layer = tracing_subscriber::fmt::layer();

    #[cfg(feature = "tracing-otlp")]
    {
        let otlp_layer = skeg_server::tracing_otlp::install()?;
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otlp_layer)
            .init();
    }
    #[cfg(not(feature = "tracing-otlp"))]
    {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", HELP);
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("skeg-resp3 {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    init_tracing()?;

    let cfg = Config::parse(args.into_iter());
    if cfg.speed {
        // Latch the opt-in into skeg-vector's process-wide flag before any
        // shard runs a search. Failure means it was set already (env-var
        // fallback) - benign.
        let _ = skeg_vector::set_speed_enabled(true);
        tracing::info!("--speed: early-termination enabled (recall@10 -0.3 to -0.7%, +40-60% QPS)");
    }
    if cfg.workers > 0 {
        tracing::info!(
            "--workers {}: VSEARCH dispatched to tokio blocking pool",
            cfg.workers
        );
    }
    if cfg.tier_mmap {
        tracing::info!("--tier-mmap: TurboQuant codes memory-mapped via tier.cache.bin");
    }
    if cfg.graph_mmap {
        tracing::info!("--graph-mmap: graph.vmn Node array memory-mapped");
    }
    let data_dir = std::path::Path::new(&cfg.data_dir);
    let server = if cfg.serve {
        tracing::info!(
            "RESP3 serve mode (read-only) over {}, tier {:?}",
            cfg.data_dir,
            cfg.tier
        );
        Server::bind_serve_full_mmap(
            &cfg.addr,
            data_dir,
            cfg.tier,
            cfg.workers,
            cfg.tier_mmap,
            cfg.graph_mmap,
        )
        .await?
    } else {
        let n_shards = skeg_platform::num_performance_cores();
        Server::bind_full_mmap(
            &cfg.addr,
            data_dir,
            n_shards,
            cfg.workers,
            cfg.tier_mmap,
            cfg.graph_mmap,
            cfg.tier,
        )
        .await?
    };
    tracing::info!("skeg-resp3 listening on {}", server.local_addr()?);
    let run_result = server.run_resp3().await;
    #[cfg(feature = "tracing-otlp")]
    skeg_server::tracing_otlp::shutdown();
    run_result?;
    Ok(())
}

fn parse_tier(arg: Option<&String>) -> QuantKind {
    // Default (no --tier) is tq2: recall ~1.0 across 100-1024d on real embeddings
    // (gate 2026-06), sub-int8 RAM - the recommended sweet spot. Pass --tier int8
    // for the full-fidelity tier.
    match arg.map(String::as_str) {
        None | Some("turboquant-2") | Some("tq2") => QuantKind::TurboQuant { bits: 2 },
        Some("int8") => QuantKind::Int8,
        Some("pq") => QuantKind::Pq { m: 128, k: 256 },
        Some(s) if s.starts_with("pq:") => {
            let mut parts = s[3..].split(':');
            let m = parts.next().and_then(|x| x.parse().ok()).unwrap_or(128);
            let k = parts.next().and_then(|x| x.parse().ok()).unwrap_or(256);
            QuantKind::Pq { m, k }
        }
        Some("turboquant-1") | Some("tq1") => QuantKind::TurboQuant { bits: 1 },
        Some("turboquant-4") | Some("tq4") => QuantKind::TurboQuant { bits: 4 },
        Some(other) => {
            tracing::warn!("unknown --tier '{other}', using the default tq2");
            QuantKind::TurboQuant { bits: 2 }
        }
    }
}

struct Config {
    addr: String,
    data_dir: String,
    serve: bool,
    tier: QuantKind,
    speed: bool,
    workers: usize,
    tier_mmap: bool,
    graph_mmap: bool,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Config {
        let mut cfg = Config {
            addr: std::env::var("SKEG_RESP3_ADDR").unwrap_or_else(|_| "127.0.0.1:6379".to_owned()),
            data_dir: std::env::var("SKEG_DATA_DIR").unwrap_or_else(|_| "./data".to_owned()),
            serve: false,
            tier: QuantKind::Int8,
            speed: matches!(
                std::env::var("SKEG_SPEED").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            ),
            workers: std::env::var("SKEG_WORKERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            tier_mmap: matches!(
                std::env::var("SKEG_TIER_MMAP").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            ),
            graph_mmap: matches!(
                std::env::var("SKEG_GRAPH_MMAP").as_deref(),
                Ok("1") | Ok("true") | Ok("on")
            ),
        };
        let args: Vec<String> = args.collect();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--addr" => {
                    if let Some(v) = args.get(i + 1) {
                        cfg.addr.clone_from(v);
                    }
                    i += 2;
                }
                "--data-dir" => {
                    if let Some(v) = args.get(i + 1) {
                        cfg.data_dir.clone_from(v);
                    }
                    i += 2;
                }
                "--mode" => {
                    cfg.serve = args.get(i + 1).map(String::as_str) == Some("serve");
                    i += 2;
                }
                "--tier" => {
                    cfg.tier = parse_tier(args.get(i + 1));
                    i += 2;
                }
                "--speed" => {
                    cfg.speed = true;
                    i += 1;
                }
                "--workers" => {
                    if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                        cfg.workers = v;
                    }
                    i += 2;
                }
                "--tier-mmap" => {
                    cfg.tier_mmap = true;
                    i += 1;
                }
                "--graph-mmap" => {
                    cfg.graph_mmap = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }
        cfg
    }
}
