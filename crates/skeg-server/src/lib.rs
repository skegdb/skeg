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

pub mod admission;
pub mod bind_policy;
pub mod catalog_intent;
pub mod failpoint;
pub mod handler;
pub mod ingress;
pub mod layout_manifest;
pub mod memory;
pub mod payload;
pub mod payload_disk;
pub mod quota;
pub mod resp3_handler;
pub mod router;
pub mod shard;
pub mod tenant;
#[cfg(feature = "tracing-otlp")]
pub mod tracing_otlp;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::{future::Future, io};
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

pub use admission::{AdmissionError, Retryability};
pub use bind_policy::{ALLOW_ENV, ALLOW_FLAG, check_unauthenticated_bind};
use handler::handle_connection;
pub use ingress::{ConnectionBudget, IngressBudget, IngressCap, IngressRejected};
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

/// Time allowed for already-accepted connections to finish after shutdown.
pub const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
enum WireProtocol {
    Native,
    Resp3,
}

pub struct Server {
    listener: TcpListener,
    shards: ShardSet,
    /// The aggregate ingress budget. Built from the same `MemoryGovernor` the
    /// shard set carries, so a byte a socket holds and a byte the delta holds
    /// are counted in one place.
    ingress: Arc<IngressBudget>,
    /// Concurrent connections this listener will serve. Each one can buffer
    /// up to the frame ceiling, so an unbounded accept loop is a memory-DoS
    /// surface; both protocols hold a permit for the connection's lifetime.
    max_connections: usize,
    /// Optional multi-tenant backend. `None` keeps single-tenant
    /// semantics; wiring an `Arc<dyn TenantBackend>` enables RESP3
    /// AUTH + per-tenant key scoping on this listener.
    tenant_backend: Option<Arc<dyn TenantBackend>>,
}

impl Server {
    /// The shard set this server serves from. Exists so a test can assert
    /// what a SERVE-mode bind actually opened - the shard-count bug lived in
    /// the bind, so a test that never crosses the bind cannot catch it.
    #[must_use]
    pub fn shards(&self) -> &ShardSet {
        &self.shards
    }

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
    /// `workers == 0` (default) keeps VSEARCH inline on the shard thread.
    /// `workers > 0` creates dedicated bounded
    /// VSEARCH workers per shard so KV ops do not queue behind vector searches.
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
            data_dir, n_shards, false, tier, workers, mmap_tier, mmap_graph,
        )?;
        let listener = TcpListener::bind(addr).await?;
        let ingress = Arc::new(ingress::IngressBudget::from_env(
            Arc::clone(shards.memory()),
            resp3_handler::MAX_CONN_BUFFER as u64,
        ));
        ingress.register_metrics();
        Ok(Self {
            listener,
            shards,
            ingress,
            max_connections: max_connections_from_env(),
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

    /// Serve with the ingress budget supplied, instead of the one derived from
    /// this process's memory governor.
    ///
    /// Exists so a test can bind a real listener against a budget it chose -
    /// a cap of a few hundred kilobytes, refusals in milliseconds - rather
    /// than by arranging for the machine to run out of memory. Same seam, and
    /// the same reason, as `ShardSet::open_full_with_memory`.
    #[must_use]
    pub fn with_ingress_budget(mut self, budget: Arc<IngressBudget>) -> Self {
        // The gauges follow the budget the server actually admits against,
        // not the one it was built with; otherwise a test that injects a
        // budget scrapes numbers belonging to a budget nothing uses.
        budget.register_metrics();
        self.ingress = budget;
        self
    }

    /// The ingress budget this server admits connections against.
    #[must_use]
    pub fn ingress(&self) -> &Arc<IngressBudget> {
        &self.ingress
    }

    /// Serve at most `n` concurrent connections, instead of the figure
    /// `SKEG_MAX_CONNECTIONS` supplies.
    ///
    /// A test seam, for the same reason as [`Server::with_ingress_budget`]:
    /// the alternative is an environment variable, which is process-wide and
    /// therefore shared by every test in the binary.
    #[must_use]
    pub fn with_max_connections(mut self, n: usize) -> Self {
        self.max_connections = n.max(1);
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
        // and the first query blocks until `run()` starts, a phantom ~8s stall
        // at 500k. Bind-after-open makes `wait_tcp` mean "ready".
        let shards = {
            // The count comes from the DATA, never from a constant: this
            // used to be a hardcoded 1, so a read-only replica of an
            // eight-shard set silently served an eighth of it.
            // No `.max(1)`: an empty directory is not a one-shard
            // replica. A layout that cannot be established is a refusal
            // to start, not a guess.
            let layout = crate::layout_manifest::LayoutManifest::open_or_migrate(
                data_dir,
                crate::layout_manifest::OpenMode::ReadOnly,
            )?;
            let n = layout.shard_count().get();
            tracing::info!("serve mode: {n} shard(s) declared by the store");
            ShardSet::open_mode_full_mmap(data_dir, n, true, tier, workers, mmap_tier, mmap_graph)?
        };
        let listener = TcpListener::bind(addr).await?;
        let ingress = Arc::new(ingress::IngressBudget::from_env(
            Arc::clone(shards.memory()),
            resp3_handler::MAX_CONN_BUFFER as u64,
        ));
        ingress.register_metrics();
        Ok(Self {
            listener,
            shards,
            ingress,
            max_connections: max_connections_from_env(),
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
        self.run_until(std::future::pending::<io::Result<()>>())
            .await
    }

    /// Run the binary protocol until `shutdown` resolves, then stop accepting,
    /// drain connections and execute the durable shard barrier.
    pub async fn run_until<F>(self, shutdown: F) -> io::Result<()>
    where
        F: Future<Output = io::Result<()>>,
    {
        self.run_until_with_timeout(shutdown, shutdown_timeout_from_env())
            .await
    }

    /// [`Server::run_until`] with an explicit connection-drain deadline.
    pub async fn run_until_with_timeout<F>(self, shutdown: F, timeout: Duration) -> io::Result<()>
    where
        F: Future<Output = io::Result<()>>,
    {
        self.run_protocol_until(WireProtocol::Native, shutdown, timeout)
            .await
    }

    /// Like `run`, but speaks RESP3 (Redis wire) on the listener instead of
    /// the skeg binary protocol. Same shard set, same storage, different
    /// encoding.
    ///
    /// # Errors
    ///
    /// Returns the first error from `TcpListener::accept`.
    pub async fn run_resp3(self) -> std::io::Result<()> {
        self.run_resp3_until(std::future::pending::<io::Result<()>>())
            .await
    }

    /// RESP3 counterpart of [`Server::run_until`].
    pub async fn run_resp3_until<F>(self, shutdown: F) -> io::Result<()>
    where
        F: Future<Output = io::Result<()>>,
    {
        self.run_resp3_until_with_timeout(shutdown, shutdown_timeout_from_env())
            .await
    }

    /// [`Server::run_resp3_until`] with an explicit connection-drain deadline.
    pub async fn run_resp3_until_with_timeout<F>(
        self,
        shutdown: F,
        timeout: Duration,
    ) -> io::Result<()>
    where
        F: Future<Output = io::Result<()>>,
    {
        self.run_protocol_until(WireProtocol::Resp3, shutdown, timeout)
            .await
    }

    async fn run_protocol_until<F>(
        self,
        protocol: WireProtocol,
        shutdown: F,
        timeout: Duration,
    ) -> io::Result<()>
    where
        F: Future<Output = io::Result<()>>,
    {
        let Self {
            listener,
            shards,
            ingress,
            max_connections,
            tenant_backend,
        } = self;
        let fds = raise_descriptor_limit();
        info!(
            addr = ?listener.local_addr()?,
            n_shards = shards.n_shards(),
            tenant = tenant_backend.is_some(),
            max_fds = fds,
            protocol = match protocol { WireProtocol::Native => "native", WireProtocol::Resp3 => "resp3" },
            "server listening"
        );
        let conn_limit = std::sync::Arc::new(tokio::sync::Semaphore::new(max_connections));
        let fp_key: Arc<str> = Arc::from(listener.local_addr()?.port().to_string());
        let mut connections = tokio::task::JoinSet::new();
        let mut failures = Vec::new();
        tokio::pin!(shutdown);

        'accept: loop {
            while let Some(done) = connections.try_join_next() {
                if let Err(e) = done {
                    failures.push(format!("connection task failed: {e}"));
                }
            }
            // Take capacity before accept. This preserves the old bounded
            // backlog without parking an already-admitted socket, and the
            // select keeps a saturated listener responsive to shutdown.
            let permit = tokio::select! {
                signal = &mut shutdown => {
                    if let Err(e) = signal { failures.push(format!("shutdown signal failed: {e}")); }
                    break 'accept;
                }
                permit = conn_limit.clone().acquire_owned() => {
                    permit.expect("connection semaphore is never closed")
                }
            };
            let (stream, _) = tokio::select! {
                signal = &mut shutdown => {
                    if let Err(e) = signal { failures.push(format!("shutdown signal failed: {e}")); }
                    break 'accept;
                }
                accepted = listener.accept() => match accepted {
                    Ok(pair) => pair,
                    Err(e) => {
                        failures.push(format!("listener accept failed: {e}"));
                        break 'accept;
                    }
                }
            };
            let refusal_wire = match protocol {
                WireProtocol::Native => RefusalWire::Native,
                WireProtocol::Resp3 => RefusalWire::Resp3,
            };
            let Some((budget, stream)) = admit_or_refuse(&ingress, stream, refusal_wire).await
            else {
                drop(permit);
                continue;
            };
            tune_socket(&stream);
            let shards = shards.clone();
            let backend = tenant_backend.clone();
            let fp_key = Arc::clone(&fp_key);
            connections.spawn(async move {
                let _permit = permit;
                match protocol {
                    WireProtocol::Native => {
                        handle_connection(stream, shards, budget, fp_key).await;
                    }
                    WireProtocol::Resp3 => {
                        handle_connection_resp3(stream, shards, backend, budget, fp_key).await;
                    }
                }
            });
        }

        // Dropping the listener is the accept barrier. Existing tasks retain
        // only their streams and shard handles.
        drop(listener);
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::ShutdownStarted);
        let drain = async {
            while let Some(done) = connections.join_next().await {
                if let Err(e) = done {
                    failures.push(format!("connection task failed: {e}"));
                }
            }
        };
        if tokio::time::timeout(timeout, drain).await.is_err() {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::ShutdownConnectionTimeouts);
            let remaining = connections.len();
            connections.abort_all();
            while connections.join_next().await.is_some() {}
            failures.push(format!(
                "connection drain deadline expired after {} ms with {remaining} task(s)",
                timeout.as_millis()
            ));
        }

        if let Err(e) = shards.shutdown().await {
            failures.extend(e.failures().iter().cloned());
        }
        if failures.is_empty() {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::ShutdownCompleted);
            Ok(())
        } else {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::ShutdownFailures);
            Err(io::Error::other(failures.join("; ")))
        }
    }
}

/// Which wire an accept-time refusal has to be spelled in.
///
/// The two protocols agree on nothing except that the peer must be told, so
/// the refusal is composed here rather than in either handler.
#[derive(Clone, Copy)]
enum RefusalWire {
    Resp3,
    Native,
}

/// Descriptor headroom, the way every production database handles it: the
/// default soft limit is a shell convention (256 on macOS, 1024 on many Linux
/// distros), not a capacity decision, and this engine holds one descriptor per
/// vlog segment and per vindex segment file - plus one per connection, which
/// is what makes it a listener's business and not only the store's. Raise it
/// toward the hard limit at boot; a refusal is logged, not hidden, so an
/// operator can raise the hard limit themselves.
fn raise_descriptor_limit() -> u64 {
    let want_fds = std::env::var("SKEG_MAX_FDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(65_536);
    let fds = skeg_platform::raise_fd_limit(want_fds);
    if fds < want_fds {
        warn!(
            soft = fds,
            wanted = want_fds,
            "file-descriptor limit below the target; raise the hard limit \
             (ulimit -n) if the store grows past a few hundred segments"
        );
    }
    fds
}

/// Take the connection's floor, or refuse the connection by name.
///
/// Returns `None` when the peer was refused; the socket is answered and closed
/// on a task of its own so a peer that never reads the answer cannot wedge the
/// accept loop. That task holds one socket and one short string for at most a
/// second, which is less than the connection would have cost admitted.
///
/// Answered BY NAME rather than shut in the peer's face: a connection dropped
/// without a word looks to a client like a network fault, and the retry it
/// schedules is the one thing a server out of room cannot afford.
async fn admit_or_refuse(
    ingress: &Arc<IngressBudget>,
    stream: TcpStream,
    wire: RefusalWire,
) -> Option<(ConnectionBudget, TcpStream)> {
    match ingress.try_accept() {
        Ok(budget) => Some((budget, stream)),
        Err(e) => {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::IngressRefusedAccept);
            warn!("ingress refused a connection: {e}");
            let admission = crate::admission::AdmissionError::from(e);
            tokio::spawn(async move {
                let mut stream = stream;
                let bytes = match wire {
                    RefusalWire::Resp3 => format!("-{}\r\n", admission.wire_message()).into_bytes(),
                    // req_id 0: there is no request yet, and inventing one
                    // would make a client match this to something it sent.
                    // The code byte says retryable and the message says why:
                    // this refusal happens before any request exists, so
                    // there is no negotiated version to gate the byte on and
                    // none is asked for.
                    RefusalWire::Native => {
                        skeg_proto::encode_err(0, admission.code(), &admission.to_string()).to_vec()
                    }
                };
                let write = async {
                    use tokio::io::AsyncWriteExt;
                    let _ = stream.write_all(&bytes).await;
                    let _ = stream.shutdown().await;
                };
                let _ = tokio::time::timeout(Duration::from_secs(1), write).await;
            });
            None
        }
    }
}

/// Concurrent connections a listener serves unless told otherwise.
///
/// The default is a capacity decision, not a shell convention: each connection
/// can hold its floor of the ingress budget for as long as it is open.
pub const DEFAULT_MAX_CONNECTIONS: usize = 1024;

/// `SKEG_MAX_CONNECTIONS`, or the default. An unparseable or zero value is
/// "not set": a typo must not become a server that accepts nothing.
fn max_connections_from_env() -> usize {
    std::env::var("SKEG_MAX_CONNECTIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_CONNECTIONS)
}

fn shutdown_timeout_from_env() -> Duration {
    std::env::var("SKEG_SHUTDOWN_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .map_or(DEFAULT_SHUTDOWN_TIMEOUT, Duration::from_millis)
}

/// Wait for Ctrl-C or SIGTERM. The signal stream is installed before waiting,
/// so a Unix termination request is consumed by the graceful lifecycle rather
/// than by the kernel's default immediate exit.
pub async fn shutdown_signal() -> io::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result,
            _ = terminate.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c().await
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
