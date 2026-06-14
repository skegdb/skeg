#![deny(unsafe_code)]

//! Multi-tenant `skeg-server` binary.
//!
//! Wraps the `skeg-server` engine with a `TenantBackend` backed by the
//! `skeg-tenant` crate. The CLI surface is identical to the single-tenant
//! binary plus two extra flags:
//!
//! - `--tenant-auth <path>` enables tenant resolution against an
//!   `auth.kdb` on disk
//! - `--tenant-strict` rejects anonymous HELLO 3 (no AUTH)

use skeg_server::Server;
use skeg_vector::QuantKind;
use tracing_subscriber::EnvFilter;

use skeg_server_tenant::AuthStoreBackend;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[allow(non_upper_case_globals, unsafe_code)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: Option<&'static core::ffi::c_char> = Some(
    // SAFETY: C string literal, NUL-terminated, jemalloc reads as `const char *`.
    unsafe { &*c"background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0".as_ptr() },
);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cfg = Config::parse(std::env::args().skip(1));
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
        )
        .await?
    };

    let server = if let Some(path) = cfg.tenant_auth.as_ref() {
        if cfg.tenant_strict {
            tracing::info!("--tenant-auth {} (strict): anonymous HELLO rejected", path);
        } else {
            tracing::info!(
                "--tenant-auth {} (lenient): anonymous HELLO maps to ZERO",
                path
            );
        }
        let backend = AuthStoreBackend::open(path, cfg.tenant_strict)?;
        server.with_tenant_backend(backend)
    } else {
        server
    };

    tracing::info!("skeg-server (tenant) listening on {}", server.local_addr()?);
    server.run_resp3().await?;
    Ok(())
}

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

struct Config {
    addr: String,
    data_dir: String,
    serve: bool,
    tier: QuantKind,
    workers: usize,
    tier_mmap: bool,
    graph_mmap: bool,
    tenant_auth: Option<String>,
    tenant_strict: bool,
}

impl Config {
    fn parse(args: impl Iterator<Item = String>) -> Config {
        let mut cfg = Config {
            addr: std::env::var("SKEG_RESP3_ADDR").unwrap_or_else(|_| "127.0.0.1:6379".to_owned()),
            data_dir: std::env::var("SKEG_DATA_DIR").unwrap_or_else(|_| "./data".to_owned()),
            serve: false,
            tier: QuantKind::Int8,
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
            tenant_auth: std::env::var("SKEG_TENANT_AUTH").ok(),
            tenant_strict: matches!(
                std::env::var("SKEG_TENANT_STRICT").as_deref(),
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
                "--tenant-auth" => {
                    if let Some(v) = args.get(i + 1) {
                        cfg.tenant_auth = Some(v.clone());
                    }
                    i += 2;
                }
                "--tenant-strict" => {
                    cfg.tenant_strict = true;
                    i += 1;
                }
                _ => i += 1,
            }
        }
        cfg
    }
}
