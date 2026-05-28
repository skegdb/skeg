#![deny(unsafe_code)]

use skeg_server::Server;
use skeg_vector::QuantKind;
use tracing_subscriber::EnvFilter;

/// jemalloc as the global allocator. The system allocator on macOS keeps
/// freed pages mapped - they stay in RSS until memory pressure - so the
/// transient buffers a streaming-insert consolidation allocates never leave
/// the process footprint. jemalloc returns decayed pages to the OS on a
/// timer, which is the foundation of post-Q10 Step 1 (Consolidation Hygiene).
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc tuning, read once at allocator init via the `malloc_conf` symbol.
///
/// `background_thread:true` reclaims decayed pages on a dedicated thread, so
/// an idle server returns memory without an allocation to drive the decay;
/// `dirty_decay_ms:1000` returns dirty pages to the OS ~1 s after they fall
/// idle (default is 10 s), keeping streaming-insert consolidation churn from
/// piling up in RSS; `muzzy_decay_ms:0` purges the intermediate state at once.
///
/// jemalloc reads `malloc_conf` as a C `const char *`, so the exported symbol
/// must be a thin pointer to a NUL-terminated string (`Option<&c_char>` has
/// exactly that layout). `#[export_name]` and the `&u8` -> `&c_char`
/// reinterpret are an allocator-configuration ABI hook with no memory-safety
/// logic, so the crate-wide `deny(unsafe_code)` is relaxed for this one item.
#[allow(non_upper_case_globals, unsafe_code)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: Option<&'static core::ffi::c_char> = Some(
    // SAFETY: the C string literal is NUL-terminated, so jemalloc reads a
    // valid C string; dereferencing its pointer yields the `&c_char` that
    // jemalloc's `const char *` ABI expects.
    unsafe { &*c"background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0".as_ptr() },
);

const HELP: &str = concat!(
    "\
skeg ",
    env!("CARGO_PKG_VERSION"),
    "

USAGE:
    skeg [OPTIONS]

OPTIONS:
    --addr <HOST:PORT>     Listen address. Default 127.0.0.1:7379. Env: SKEG_ADDR.
    --data-dir <PATH>      Data directory. Default ./data. Env: SKEG_DATA_DIR.
    --mode <MODE>          'rw' (default) or 'serve' (read-only, mmap tier).
    --tier <KIND>          Quantizer for the serve tier:
                             int8 (default) | pq | pq:M:K |
                             turboquant-1 | turboquant-2 | turboquant-4
                             (aliases: tq1, tq2, tq4)
    --speed                Opt-in early-termination in greedy walk
                             (-0.3 to -0.7% recall@10, +40-60% QPS).
                             Also: SKEG_SPEED=1.
    --workers <N>          Dispatch SKEG.VSEARCH to a worker pool (N threads).
                             0 (default) = inline on shard. Also: SKEG_WORKERS.
    --tier-mmap            mmap the TurboQuant tier (tier.cache.bin) instead of
                             holding it in RAM. Also: SKEG_TIER_MMAP=1.
    --graph-mmap           mmap the Vamana graph Node array. Also: SKEG_GRAPH_MMAP=1.
    --metrics-port <PORT>  Bind a Prometheus /metrics exporter on 127.0.0.1:PORT.
                            Requires the binary to be built with the
                            `metrics-http` cargo feature. Default: off.
                            Env: SKEG_METRICS_PORT.
    -h, --help             Print this help.
    -V, --version          Print the version.

PROTOCOL: native binary on the listen port (use skeg-resp3 for RESP3 / Redis).
DOCS:     https://github.com/skegdb/skeg
"
);

/// Spawn the Prometheus exporter on `127.0.0.1:port`.
///
/// Feature-gated: when the binary is built without `metrics-http`, the
/// function logs a warning and returns without spawning anything · the
/// flag is parsed in either build but only effective when the feature
/// is enabled.
fn spawn_metrics_exporter(port: u16) {
    #[cfg(feature = "metrics-http")]
    {
        let addr: std::net::SocketAddr = ([127, 0, 0, 1], port).into();
        match skeg_telemetry::http::spawn(addr) {
            Ok(_handle) => {
                tracing::info!(
                    "--metrics-port {port}: Prometheus exporter on http://{addr}/metrics"
                );
            }
            Err(e) => {
                tracing::warn!("--metrics-port {port}: exporter failed to bind: {e}");
            }
        }
    }
    #[cfg(not(feature = "metrics-http"))]
    {
        let _ = port;
        tracing::warn!(
            "--metrics-port set but binary was built without the `metrics-http` feature; \
             use `STATS` over RESP3 instead, or rebuild with `--features metrics-http`."
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", HELP);
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("skeg {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cfg = Config::parse(args.into_iter());
    if cfg.speed {
        // Latch the opt-in into skeg-vector's process-wide flag. Has to
        // happen before the first search, which is what `Server::bind*`
        // can trigger via recovery; doing it here in main is the earliest
        // safe point. Failure means it was already set (e.g. from the
        // env-var fallback path) - benign.
        let _ = skeg_vector::set_speed_enabled(true);
        tracing::info!("--speed: early-termination enabled (recall@10 -0.3 to -0.7%, +40-60% QPS)");
    }
    if cfg.workers > 0 {
        tracing::info!(
            "--workers {}: VSEARCH dispatched to tokio blocking pool (KV ops stay inline)",
            cfg.workers
        );
    }
    if cfg.tier_mmap {
        tracing::info!(
            "--tier-mmap: TurboQuant codes persisted to tier.cache.bin and memory-mapped"
        );
    }
    if cfg.graph_mmap {
        tracing::info!(
            "--graph-mmap: graph.vmn opened as MappedFile, Node array reinterpreted from mmap"
        );
    }
    if let Some(port) = cfg.metrics_port {
        spawn_metrics_exporter(port);
    }
    let data_dir = std::path::Path::new(&cfg.data_dir);
    let server = if cfg.serve {
        tracing::info!(
            "serve mode (read-only) over {}, tier {:?}",
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
        )
        .await?
    };
    tracing::info!("skeg listening on {}", server.local_addr()?);
    server.run().await?;
    Ok(())
}

/// Parse a `--tier` argument:
/// - `int8` (default): 8-bit symmetric, max recall
/// - `pq` / `pq:M:K`: Product Quantization (default M=128 K=256)
/// - `turboquant-N` where N in {1, 2, 4}: TurboQuant data-oblivious tier
///   (`turboquant-1` matches the PQ-128 footprint without k-means training)
///
/// An unknown value falls back to int8 with a warning.
fn parse_tier(arg: Option<&String>) -> QuantKind {
    match arg.map(String::as_str) {
        None | Some("int8") => QuantKind::Int8,
        Some("pq") => QuantKind::Pq { m: 128, k: 256 },
        Some(s) if s.starts_with("pq:") => {
            let mut parts = s[3..].split(':');
            let m = parts.next().and_then(|x| x.parse().ok()).unwrap_or(128);
            let k = parts.next().and_then(|x| x.parse().ok()).unwrap_or(256);
            QuantKind::Pq { m, k }
        }
        Some("turboquant-1") | Some("tq1") => QuantKind::TurboQuant { bits: 1 },
        Some("turboquant-2") | Some("tq2") => QuantKind::TurboQuant { bits: 2 },
        Some("turboquant-4") | Some("tq4") => QuantKind::TurboQuant { bits: 4 },
        Some(other) => {
            tracing::warn!("unknown --tier '{other}', using int8");
            QuantKind::Int8
        }
    }
}

/// Server configuration: CLI flags override the `SKEG_*` environment vars.
struct Config {
    addr: String,
    data_dir: String,
    serve: bool,
    /// Tier-1 quantisation for serve mode (ignored in read-write mode).
    tier: QuantKind,
    /// Opt-in early-termination on the Vamana graph walk: trades
    /// 0.3-0.7% recall@10 for +40-60% QPS. Maps to `SKEG_SPEED=1` so
    /// skeg-vector picks it up at first search.
    speed: bool,
    /// Opt-in VSEARCH worker pool (Q11, Tier 2 of `optimizations/PLAN.md`).
    /// `0` = inline VSEARCH on the shard thread (default; matches the public
    /// bench numbers). `> 0` = dispatch VSEARCH to tokio's blocking pool so
    /// KV ops do not queue behind multi-ms searches. The integer value is
    /// informational today; a future dedicated pool will honour it.
    workers: usize,
    /// Opt-in TurboQuant tier paging (Position 2 of the VeloANN paging
    /// discussion, OBSERVATIONS 2026-05-21). When set, the TurboQuant
    /// `codes` buffer is persisted to `tier.cache.bin` at open and
    /// memory-mapped: the OS page cache can reclaim tier pages under
    /// pressure instead of pushing anonymous memory to swap. `int8` and
    /// `pq` tiers are unaffected for now. Env var `SKEG_TIER_MMAP`.
    tier_mmap: bool,
    /// Opt-in graph paging (Position 2.5). When set, `graph.vmn` is opened
    /// as a `MappedFile` and the Node array is reinterpreted directly from
    /// the mmap'd bytes - no per-Node parsing into `Vec<Node>` at open,
    /// and OS page cache can reclaim graph pages under pressure. Combines
    /// with `--tier-mmap` to make the whole disk index paginable.
    /// Env var `SKEG_GRAPH_MMAP`.
    graph_mmap: bool,
    /// Opt-in Prometheus metrics HTTP exporter. When set, a tiny HTTP
    /// server runs on `127.0.0.1:<port>` and serves `/metrics` in the
    /// Prometheus text format from the same telemetry counters that the
    /// RESP3 `STATS` command reads. Only available when the binary is
    /// built with `--features metrics-http`. Env var `SKEG_METRICS_PORT`.
    metrics_port: Option<u16>,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Config {
        let mut cfg = Config {
            addr: std::env::var("SKEG_ADDR").unwrap_or_else(|_| "127.0.0.1:7379".to_owned()),
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
            metrics_port: std::env::var("SKEG_METRICS_PORT")
                .ok()
                .and_then(|v| v.parse().ok()),
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
                "--metrics-port" => {
                    if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                        cfg.metrics_port = Some(v);
                    }
                    i += 2;
                }
                _ => i += 1,
            }
        }
        cfg
    }
}
