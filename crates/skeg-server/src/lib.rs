#![deny(unsafe_code)]
// Server::bind_* funcs take many tuning knobs (workers, mmap flags,
// shard count, ...). Wrapping them in a config struct would push the
// complexity to every call site without a real win.
#![allow(clippy::too_many_arguments)]

//! `skeg-server` - TCP server library.
//!
//! Ships single-tenant by default. A separate crate (see the
//! `tenant` module docs) can install a multi-tenant layer at runtime
//! via [`Server::with_tenant_backend`].

pub mod handler;
pub mod payload;
pub mod quota;
pub mod resp3_handler;
pub mod shard;
pub mod tenant;
#[cfg(feature = "tracing-otlp")]
pub mod tracing_otlp;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

use handler::handle_connection;
pub use quota::{TenantLimits, TenantQos, TenantVectorQuota};
use resp3_handler::handle_connection_resp3;
use shard::ShardSet;
pub use shard::{ControlHandle, IndexStat};
use skeg_vector::QuantKind;
pub use tenant::{
    Admission, AdmitGuard, AdmitRejected, AnonymousPolicy, CommandKind, QuotaAdminError,
    TenantBackend, TenantId,
};

/// Default quantiser tier for the read-write path and the CLI. TurboQuant 2-bit:
/// the product's tier of record (data-oblivious, 4x smaller than int8), cheap to
/// rebuild since the tier build parallelises across cores. Requires `dim % 4 == 0`.
pub const DEFAULT_RW_TIER: QuantKind = QuantKind::TurboQuant { bits: 2 };

pub struct Server {
    listener: TcpListener,
    shards: ShardSet,
    /// Optional multi-tenant backend. `None` keeps single-tenant
    /// semantics; wiring an `Arc<dyn TenantBackend>` enables RESP3
    /// AUTH + per-tenant key scoping on this listener.
    tenant_backend: Option<Arc<dyn TenantBackend>>,
}

impl Server {
    /// Bind the server to `addr` with data sharded under `data_dir`.
    ///
    /// The shard count equals the number of performance cores.
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or a shard cannot start.
    pub async fn bind(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
    ) -> std::io::Result<Self> {
        let n_shards = skeg_platform::num_performance_cores();
        Self::bind_full(addr, data_dir, n_shards, 0, false, DEFAULT_RW_TIER).await
    }

    /// Bind the server with an explicit shard count and worker-pool size.
    ///
    /// `workers == 0` (default) keeps VSEARCH inline on the shard thread
    /// (Personal AI default). `workers > 0` dispatches VSEARCH to a tokio
    /// blocking pool so KV ops do not queue behind multi-ms vector searches
    /// (multi-tenant pattern, opt-in via `--workers N` on the CLI).
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or a shard cannot start.
    pub async fn bind_with_shards(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
        n_shards: usize,
        workers: usize,
    ) -> std::io::Result<Self> {
        Self::bind_full(addr, data_dir, n_shards, workers, false, DEFAULT_RW_TIER).await
    }

    /// Full-knob constructor for the read-write path: shard count, worker
    /// pool, and the opt-in `mmap_tier` flag (see `--tier-mmap` in the
    /// server CLI). Other entry points delegate here with defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or a shard cannot start.
    pub async fn bind_full(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
        n_shards: usize,
        workers: usize,
        mmap_tier: bool,
        tier: QuantKind,
    ) -> std::io::Result<Self> {
        Self::bind_full_mmap(addr, data_dir, n_shards, workers, mmap_tier, false, tier).await
    }

    /// All-knobs constructor for the read-write path. Adds `mmap_graph`
    /// to [`bind_full`](Self::bind_full).
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or a shard cannot start.
    pub async fn bind_full_mmap(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
        n_shards: usize,
        workers: usize,
        mmap_tier: bool,
        mmap_graph: bool,
        tier: QuantKind,
    ) -> std::io::Result<Self> {
        // Recover shards before binding so the port only opens once queries can
        // be served (see bind_serve_full_mmap for the phantom-stall rationale).
        let shards = ShardSet::open_mode_full_mmap(
            data_dir,
            n_shards,
            false,
            tier,
            workers,
            mmap_tier,
            mmap_graph,
        )?;
        let listener = TcpListener::bind(addr).await?;
        Ok(Self {
            listener,
            shards,
            tenant_backend: None,
        })
    }

    /// Install a multi-tenant backend on a server already built by one
    /// of the `bind*` constructors. Builder-style for clarity at the
    /// call site. When set, the RESP3 handler honours `HELLO 3 AUTH`
    /// and scopes KV / vector ops by tenant id.
    #[must_use]
    pub fn with_tenant_backend(mut self, backend: Arc<dyn TenantBackend>) -> Self {
        self.tenant_backend = Some(backend);
        self
    }

    /// Bind the server in serve mode: a single shard over the offline-built
    /// index at `data_dir`, read-only. Every mutation (KV and vector) is
    /// rejected; the index is served at its clean resident footprint.
    ///
    /// `data_dir` is a directory produced by `skeg-tool build`. `tier` is the
    /// tier-1 quantisation built for the served index: `QuantKind::Int8` or
    /// `QuantKind::Pq { m, k }` (smaller footprint). `workers > 0` enables
    /// the VSEARCH dispatch pool described in [`bind_with_shards`].
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or the shard cannot
    /// start.
    pub async fn bind_serve(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
        tier: QuantKind,
        workers: usize,
    ) -> std::io::Result<Self> {
        Self::bind_serve_full(addr, data_dir, tier, workers, false).await
    }

    /// Full-knob serve mode: like [`bind_serve`] plus the opt-in
    /// `mmap_tier` flag that swaps the TurboQuant codes for a
    /// memory-mapped view of `tier.cache.bin` at open.
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or the shard cannot start.
    pub async fn bind_serve_full(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
        tier: QuantKind,
        workers: usize,
        mmap_tier: bool,
    ) -> std::io::Result<Self> {
        Self::bind_serve_full_mmap(addr, data_dir, tier, workers, mmap_tier, false).await
    }

    /// All-knobs serve mode. Adds `mmap_graph` to
    /// [`bind_serve_full`](Self::bind_serve_full).
    ///
    /// # Errors
    ///
    /// Returns an error if the address cannot be bound or the shard cannot start.
    pub async fn bind_serve_full_mmap(
        addr: impl tokio::net::ToSocketAddrs,
        data_dir: &Path,
        tier: QuantKind,
        workers: usize,
        mmap_tier: bool,
        mmap_graph: bool,
    ) -> std::io::Result<Self> {
        // Open (and eagerly recover) the shard BEFORE binding, so the listen
        // port only comes up once the index is queryable. Otherwise the kernel
        // accepts connections into the backlog during the multi-second recover
        // and the first query blocks until `run()` starts — a phantom ~8s stall
        // at 500k. Bind-after-open makes `wait_tcp` mean "ready".
        let shards =
            ShardSet::open_mode_full_mmap(data_dir, 1, true, tier, workers, mmap_tier, mmap_graph)?;
        let listener = TcpListener::bind(addr).await?;
        Ok(Self {
            listener,
            shards,
            tenant_backend: None,
        })
    }

    /// Return the local address the server is listening on.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS cannot retrieve the socket address.
    pub fn local_addr(&self) -> std::io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Number of shards backing this server.
    #[must_use]
    pub fn n_shards(&self) -> usize {
        self.shards.n_shards()
    }

    /// Control-plane handle for vindex tiering (enumerate / report RAM /
    /// evict). An external policy crate attaches a background task to it; the
    /// engine provides only the mechanism.
    #[must_use]
    pub fn control_handle(&self) -> ControlHandle {
        self.shards.control_handle()
    }

    /// Accept connections and handle them until an I/O error on `accept`.
    ///
    /// # Errors
    ///
    /// Returns the first error from `TcpListener::accept`.
    pub async fn run(self) -> std::io::Result<()> {
        let Self {
            listener,
            shards,
            tenant_backend: _,
        } = self;
        info!(addr = ?listener.local_addr()?, n_shards = shards.n_shards(), "server listening (binary protocol)");
        loop {
            let (stream, _) = listener.accept().await?;
            tune_socket(&stream);
            let shards = shards.clone();
            tokio::spawn(async move {
                handle_connection(stream, shards).await;
            });
        }
    }

    /// Like `run`, but speaks RESP3 (Redis wire) on the listener instead of
    /// the skeg binary protocol. Same shard set, same storage, different
    /// encoding.
    ///
    /// # Errors
    ///
    /// Returns the first error from `TcpListener::accept`.
    pub async fn run_resp3(self) -> std::io::Result<()> {
        let Self {
            listener,
            shards,
            tenant_backend,
        } = self;
        info!(
            addr = ?listener.local_addr()?,
            n_shards = shards.n_shards(),
            tenant = tenant_backend.is_some(),
            "server listening (RESP3)"
        );
        loop {
            let (stream, _) = listener.accept().await?;
            tune_socket(&stream);
            let shards = shards.clone();
            let backend = tenant_backend.clone();
            tokio::spawn(async move {
                handle_connection_resp3(stream, shards, backend).await;
            });
        }
    }
}

/// Apply per-connection socket tuning: `TCP_NODELAY` for low-latency
/// request/reply traffic, and `SO_KEEPALIVE` + `TCP_KEEPIDLE` so a
/// half-open connection (peer dropped without FIN) gets detected
/// before the OS default of ~2h. Failures here are logged and
/// swallowed because they don't prevent the connection from working
/// (they just degrade tail-case behaviour).
fn tune_socket(stream: &TcpStream) {
    if let Err(e) = stream.set_nodelay(true) {
        warn!("set_nodelay failed: {e}");
    }
    let sock = socket2::SockRef::from(stream);
    let ka = socket2::TcpKeepalive::new()
        .with_time(Duration::from_secs(60))
        .with_interval(Duration::from_secs(10));
    if let Err(e) = sock.set_tcp_keepalive(&ka) {
        warn!("set_tcp_keepalive failed: {e}");
    }
}
