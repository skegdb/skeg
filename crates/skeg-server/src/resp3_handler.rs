//! RESP3 connection handler.
//!
//! Per-connection task that reads bytes from a TCP socket into a `FrameDecoder`,
//! parses them into `Command`s, dispatches to the `ShardSet`, encodes the
//! response back via `encode_frame`, and writes to the socket.
//!
//! Mirrors the binary-protocol `handler.rs` but speaks Redis wire (RESP2/RESP3).
//! New connections default to RESP2 until `HELLO 3` upgrades them.
//!
//! Wire commands supported in this iteration (the KV subset):
//! - `HELLO [version [AUTH user pass] [SETNAME name]]` - protocol negotiation.
//! - `PING [msg]` / `ECHO msg` - protocol-only.
//! - `GET key` / `SET key value` / `DEL key [key ...]` / `EXISTS key [key ...]`.
//! - `SELECT 0` accepted as no-op (driver compat), `SELECT N>0` rejected.
//!
//! Out of scope here (later v0.1 / v0.2): SET options (EX/PX/NX/XX), EXPIRE/TTL,
//! INFO/STATS/DBSIZE/COMMAND, SHUTDOWN, vector ops, async maintenance, AUTH model.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, warn};

use skeg_core::Durability;
use skeg_resp3::{
    Command, ConnectionState, Frame, FrameDecoder, encode_frame, handle_echo, handle_ping,
    parse_command,
};

use crate::payload::parse_filter;
use crate::shard::ShardSet;
use crate::tenant::{Admission, AnonymousPolicy, CommandKind, TenantBackend, TenantId};

/// Format a VINDEX name with the tenant scope. `TenantId::ZERO` returns the
/// raw name (single-tenant deployments stay byte-identical to pre-tenancy).
fn scoped_vindex_name(tenant: TenantId, name: &str) -> String {
    if tenant.is_zero() {
        name.to_string()
    } else {
        format!("{tenant}::{name}")
    }
}

/// Reject the tenant-scope separator in a client-supplied index name, then
/// scope it. `::` is reserved for tenant scoping and is the only way a tenant
/// prefix enters a name; without this guard an anonymous (`ZERO`) connection
/// could pass `"<victim-tenant-hex>::idx"` and have `scoped_vindex_name` return
/// it verbatim, reaching another tenant's index (read/write/drop). Every vector
/// op scopes through this helper so none can forget the check.
fn scope_vindex_or_reject(tenant: TenantId, raw_name: &str) -> Result<String, Frame> {
    if raw_name.contains("::") {
        return Err(Frame::Error(
            "ERR VINDEX name must not contain '::' (reserved for tenant scoping)".into(),
        ));
    }
    Ok(scoped_vindex_name(tenant, raw_name))
}

/// Bytes that have been confirmed to carry the tenant scope (or to be
/// part of the anonymous `ZERO` namespace). Constructed only via
/// `scope_key`; every shard call site goes through `.as_bytes()`, so
/// a future refactor that forgets the prefix step would fail to type-
/// check rather than silently leak data between tenants. This is the
/// defense-in-depth wrapper for multi-tenancy phase 1.
#[derive(Debug, Clone)]
struct ScopedKey {
    bytes: Bytes,
    tenant: TenantId,
}

impl ScopedKey {
    fn as_bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// The owning tenant id as a `u128`, for per-tenant cache accounting. `0`
    /// for the unscoped (anonymous) default, matching `VLog`'s tenant 0 path.
    fn accounting_tenant(&self) -> u128 {
        tenant_u128(self.tenant)
    }

    /// Cheap runtime check the prefix invariant still holds. Called at
    /// shard call sites; cost is a single byte-slice compare per op.
    fn assert_invariant(&self) {
        if !self.tenant.is_zero() {
            debug_assert!(
                self.bytes.len() >= TenantId::LEN
                    && &self.bytes[..TenantId::LEN] == self.tenant.as_bytes(),
                "ScopedKey invariant violated for tenant={}: byte prefix \
                 does not match the tenant id. Bug in scope_key \
                 or someone built a ScopedKey by hand.",
                self.tenant,
            );
        }
    }
}

/// Prefix `key` with the tenant id when the connection is non-anonymous.
/// Returns the original key bytes when `tenant` is `ZERO`, so single-tenant
/// traffic keeps byte-identical wire and disk semantics.
fn scope_key(tenant: TenantId, key: &Bytes) -> ScopedKey {
    let bytes = if tenant.is_zero() {
        key.clone()
    } else {
        let mut v = Vec::with_capacity(TenantId::LEN + key.len());
        v.extend_from_slice(tenant.as_bytes());
        v.extend_from_slice(key);
        Bytes::from(v)
    };
    let k = ScopedKey { bytes, tenant };
    k.assert_invariant();
    k
}

/// Reject an anonymous (ZERO) request whose key begins with bytes that
/// match a real bound tenant id. Without this check an anon client could
/// craft `<tenant_id 16B><target_key>` to read or overwrite an
/// authenticated tenant's scoped key (TenantId::from_name is a public
/// non-secret hash). Single-tenant deployments (`tenant_backend == None`)
/// skip the check, so byte-layout stays identical to the pre-tenancy
/// path.
fn anon_key_collides_with_tenant(
    tenant: TenantId,
    key: &[u8],
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> bool {
    if !tenant.is_zero() || key.len() < TenantId::LEN {
        return false;
    }
    let Some(ctx) = ctx else {
        return false;
    };
    let mut prefix = [0u8; TenantId::LEN];
    prefix.copy_from_slice(&key[..TenantId::LEN]);
    let candidate = TenantId::from_bytes(prefix);
    if candidate.is_zero() {
        return false;
    }
    ctx.has_tenant(candidate)
}

fn anon_forgery_error() -> Frame {
    Frame::Error(
        "ERR key prefix collides with a bound tenant id; \
         authenticate with HELLO 3 AUTH to use scoped keys"
            .into(),
    )
}

const SKEG_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default durability for SET / DEL. `Kernel` survives process+kernel crash
/// without `F_FULLFSYNC` cost - same default as the binary protocol handler.
const DEFAULT_DURABILITY: Durability = Durability::Kernel;

/// Server-assigned connection id, monotonic across the process lifetime.
/// Exposed via HELLO response and (future) CLIENT ID.
static CONN_COUNTER: AtomicI64 = AtomicI64::new(1);

/// Per-connection driver. Loops until EOF / write error / fatal parse error.
pub async fn handle_connection_resp3(
    mut stream: TcpStream,
    shards: ShardSet,
    tenant_backend: Option<Arc<dyn TenantBackend>>,
) {
    let peer = stream.peer_addr().ok();
    debug!(?peer, "RESP3 connection accepted");

    let id = CONN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut state = ConnectionState::new(id);
    let mut tenant = TenantId::ZERO;
    let mut decoder = FrameDecoder::new();
    let mut out = BytesMut::with_capacity(4096);

    // Pipelined dispatch: data-plane commands from one connection run
    // concurrently (bounded window, responses emitted in submission order) so a
    // client's pipelined VSET burst feeds the vLog group-committer concurrently
    // instead of one-blob-per-200µs-timer serially. Session-state commands
    // (HELLO/AUTH/... - see `is_pipelineable`) are a barrier: all in-flight
    // responses are flushed in order before the barrier runs serially, so the
    // request/response ordering the client sees is unchanged.
    const PIPELINE_WINDOW: usize = 128;
    let mut inflight: VecDeque<tokio::task::JoinHandle<Frame>> = VecDeque::new();
    // Await the oldest in-flight command and write its response in order.
    // Returns false on write failure (caller must stop).
    macro_rules! emit_front {
        () => {{
            let mut ok = true;
            if let Some(h) = inflight.pop_front() {
                let resp = h
                    .await
                    .unwrap_or_else(|_| Frame::Error("ERR internal task failure".into()));
                out.clear();
                encode_frame(&resp, state.version, &mut out);
                ok = stream.write_all(&out).await.is_ok();
            }
            ok
        }};
    }

    'conn: loop {
        match decoder.decode() {
            Ok(Some(frame)) => match parse_command(frame) {
                Ok(cmd) if is_pipelineable(&cmd) => {
                    let (sh, be, t) = (shards.clone(), tenant_backend.clone(), tenant);
                    inflight.push_back(tokio::spawn(exec_pipelined(cmd, t, sh, be)));
                    if inflight.len() >= PIPELINE_WINDOW && !emit_front!() {
                        break 'conn;
                    }
                }
                other => {
                    // Barrier (session-state command) or parse error: drain the
                    // pipeline in order, then run this one serially.
                    while !inflight.is_empty() {
                        if !emit_front!() {
                            break 'conn;
                        }
                    }
                    let response = match other {
                        Ok(cmd) => {
                            dispatch_command(
                                cmd,
                                &mut state,
                                &mut tenant,
                                &shards,
                                tenant_backend.as_ref(),
                            )
                            .await
                        }
                        Err(e) => Frame::Error(format!("ERR {e}")),
                    };
                    out.clear();
                    encode_frame(&response, state.version, &mut out);
                    if stream.write_all(&out).await.is_err() {
                        break 'conn;
                    }
                }
            },
            Ok(None) => {
                // Buffer drained: emit the in-flight burst (bounds latency + lets
                // its payload writes group-commit together), then read more.
                while !inflight.is_empty() {
                    if !emit_front!() {
                        break 'conn;
                    }
                }
                // Pull a large chunk per syscall so a pipelined burst buffers
                // many frames at once (the default 4 KiB spare = one ~4 KiB VSET
                // frame, which would serialize the pipeline one-frame-per-read).
                decoder.buf_mut().reserve(256 * 1024);
                match stream.read_buf(decoder.buf_mut()).await {
                    Ok(0) => break,
                    Ok(_) => continue,
                    Err(e) => {
                        warn!(?peer, "read error: {e}");
                        break;
                    }
                }
            }
            Err(e) => {
                while !inflight.is_empty() {
                    if !emit_front!() {
                        break 'conn;
                    }
                }
                warn!(?peer, "RESP3 parse error: {e}");
                let err = Frame::Error(format!("ERR protocol: {e}"));
                out.clear();
                encode_frame(&err, state.version, &mut out);
                let _ = stream.write_all(&out).await;
                break;
            }
        }
    }
    // Drain anything still in flight on close.
    while !inflight.is_empty() {
        if !emit_front!() {
            break;
        }
    }

    debug!(?peer, "RESP3 connection closed");
}

/// Coarse compute cost of a command, in QoS credits. A vector search's work is
/// dominated by `l_search` (the graph walk) plus `k` (the rerank), both read
/// straight from the command args with no index lookup; every other command is
/// a flat 1. Deliberately coarse: vector dimension and tenant data size are
/// roughly constant per index and can be folded in later if a measurement shows
/// the weighting is off.
fn command_cost(cmd: &Command) -> u32 {
    match cmd {
        // args: name k l_search vector [WITHPAYLOAD] [FILTER expr]
        Command::SkegVsearch { args } => {
            let num = |i: usize| {
                args.get(i)
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0)
            };
            num(1).saturating_add(num(2)).max(1)
        }
        _ => 1,
    }
}

/// Classify a command for the admission gate ([`CommandKind`]). Grouped by
/// (resource, action); index-lifecycle ops stay individual (the RBAC target).
/// Exhaustive over `Command` so a new command must be classified here.
fn command_kind(cmd: &Command) -> CommandKind {
    match cmd {
        Command::Get { .. } | Command::Mget { .. } | Command::Exists { .. } => CommandKind::KvRead,
        Command::Set { .. }
        | Command::Mset { .. }
        | Command::Del { .. }
        | Command::Incr { .. }
        | Command::Decr { .. }
        | Command::IncrBy { .. }
        | Command::DecrBy { .. } => CommandKind::KvWrite,
        Command::SkegVsearch { .. } => CommandKind::VectorRead,
        Command::SkegVset { .. } | Command::SkegVmset { .. } | Command::SkegVdel { .. } => {
            CommandKind::VectorWrite
        }
        Command::SkegVindexCreate { .. } => CommandKind::VindexCreate,
        Command::SkegVindexDrop { .. } => CommandKind::VindexDrop,
        Command::SkegVindexConsolidate { .. } => CommandKind::VindexConsolidate,
        Command::SkegVindexList => CommandKind::VindexList,
        Command::SkegQuotaSet { .. }
        | Command::SkegQuotaGet { .. }
        | Command::SkegQosSet { .. }
        | Command::SkegQosGet { .. } => CommandKind::Admin,
        Command::Hello(_)
        | Command::Ping(_)
        | Command::Echo(_)
        | Command::Select { .. }
        | Command::SkegStats
        | Command::SkegShards
        | Command::SkegWhoami
        | Command::SkegAuth { .. }
        | Command::Unknown { .. } => CommandKind::Meta,
    }
}

/// Data-plane commands that read the session tenant but never mutate session
/// state, so they can be dispatched concurrently on one connection (see the
/// pipelined connection loop). Everything else - HELLO/AUTH (mutate tenant),
/// SELECT, index lifecycle, admin - is a serial barrier.
fn is_pipelineable(cmd: &Command) -> bool {
    // ONLY commands whose intra-connection reordering is semantically invisible.
    // The scalar KV verbs (Get/Set/Del/Incr/... ) are DELIBERATELY excluded: two
    // pipelined ops on the same key must apply in submission order (SET a;SET b,
    // INCR atomicity, SET;GET read-after-write), but concurrent tasks reach the
    // shard mailbox in scheduler order, not submission order - so they stay serial
    // barriers. The vector write path (the batching target) is keyed by a distinct
    // id per bulk-ingest row; two writes to the SAME id in one pipeline are
    // last-write-wins (an upsert, acceptable). VSEARCH is read-only.
    matches!(
        cmd,
        Command::SkegVset { .. }
            | Command::SkegVmset { .. }
            | Command::SkegVsearch { .. }
            | Command::SkegVdel { .. }
            | Command::Ping(_)
            | Command::Echo(_)
    )
}

/// Run one pipelineable command with owned inputs so it can be driven
/// concurrently with its neighbours. Admission gate + the command's handler;
/// same behaviour as the matching `dispatch_command` arm, minus the `&mut`
/// session borrow. Only ever called for [`is_pipelineable`] commands.
async fn exec_pipelined(
    cmd: Command,
    tenant: TenantId,
    shards: ShardSet,
    backend: Option<Arc<dyn TenantBackend>>,
) -> Frame {
    let _admit = match backend.as_ref() {
        None => None,
        Some(ctx) => match ctx.admit(Admission {
            tenant,
            op: command_kind(&cmd),
            cost: command_cost(&cmd),
        }) {
            Ok(guard) => Some(guard),
            Err(rejected) => return Frame::Error(rejected.message),
        },
    };
    let be = backend.as_ref();
    match cmd {
        Command::SkegVset { args } => skeg_vset(&args, &shards, tenant, be).await,
        Command::SkegVmset { args } => skeg_vmset(&args, &shards, tenant, be).await,
        Command::SkegVsearch { args } => skeg_vsearch(&args, &shards, tenant).await,
        Command::SkegVdel { args } => skeg_vdel(&args, &shards, tenant).await,
        Command::Ping(msg) => handle_ping(msg),
        Command::Echo(msg) => handle_echo(msg),
        // Unreachable: the connection loop only routes `is_pipelineable` commands
        // here, and this match covers exactly that set.
        _ => Frame::Error("ERR command not pipelineable".into()),
    }
}

async fn dispatch_command(
    cmd: Command,
    state: &mut ConnectionState,
    tenant: &mut TenantId,
    shards: &ShardSet,
    tenant_backend: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    // Per-command admission (multi-tenant QoS). Hello/SkegAuth establish or
    // change the tenant and are never gated. Single-tenant (no backend) skips
    // entirely. `_admit` is held until this function returns, so a concurrency
    // cap reserved in `admit` actually bounds the command's in-flight lifetime.
    let _admit = match (&cmd, tenant_backend) {
        (Command::Hello(_) | Command::SkegAuth { .. }, _) | (_, None) => None,
        (_, Some(ctx)) => {
            let admission = Admission {
                tenant: *tenant,
                op: command_kind(&cmd),
                cost: command_cost(&cmd),
            };
            match ctx.admit(admission) {
                Ok(guard) => Some(guard),
                Err(rejected) => return Frame::Error(rejected.message),
            }
        }
    };
    match cmd {
        Command::Hello(args) => {
            // Verify credentials when AUTH is supplied. When AUTH is
            // absent and the backend asks for `Strict`, reject the
            // connection with -NOAUTH (RESP3 standard error).
            if let Some(ctx) = tenant_backend {
                match args.auth.as_ref() {
                    Some((user, pass)) => match ctx.verify_login(user, pass.as_bytes()) {
                        Some(tid) => *tenant = tid,
                        None => {
                            return Frame::Error("WRONGPASS invalid username-password pair".into());
                        }
                    },
                    None => {
                        if matches!(ctx.anonymous_policy(), AnonymousPolicy::Strict) {
                            return Frame::Error(
                                "NOAUTH authentication required (server is in strict mode)".into(),
                            );
                        }
                    }
                }
            }
            state.apply_hello(&args, SKEG_VERSION)
        }
        Command::Ping(msg) => handle_ping(msg),
        Command::Echo(msg) => handle_echo(msg),
        Command::Get { key } => {
            kv_get(std::slice::from_ref(&key), shards, *tenant, tenant_backend).await
        }
        Command::Set { key, value } => kv_set(&[key, value], shards, *tenant, tenant_backend).await,
        Command::Del { keys } => kv_del(&keys, shards, *tenant, tenant_backend).await,
        Command::Exists { keys } => kv_exists(&keys, shards, *tenant, tenant_backend).await,
        Command::Mget { keys } => kv_mget(&keys, shards, *tenant, tenant_backend).await,
        Command::Mset { pairs } => {
            let args: Vec<Bytes> = pairs.into_iter().flat_map(|(k, v)| [k, v]).collect();
            kv_mset(&args, shards, *tenant, tenant_backend).await
        }
        Command::Incr { key } => {
            kv_incr_by(
                std::slice::from_ref(&key),
                shards,
                1,
                *tenant,
                tenant_backend,
            )
            .await
        }
        Command::Decr { key } => {
            kv_incr_by(
                std::slice::from_ref(&key),
                shards,
                -1,
                *tenant,
                tenant_backend,
            )
            .await
        }
        Command::IncrBy { key, delta } => {
            kv_incrby_apply(&key, delta, shards, *tenant, tenant_backend).await
        }
        Command::DecrBy { key, delta } => {
            // DECRBY semantics: subtract delta. Negating without underflow
            // check would silently wrap on i64::MIN; reject explicitly.
            let signed = match delta.checked_neg() {
                Some(v) => v,
                None => return Frame::Error("ERR value out of range".into()),
            };
            kv_incrby_apply(&key, signed, shards, *tenant, tenant_backend).await
        }
        Command::Select { db } => kv_select_db(db),
        Command::SkegStats => skeg_stats(shards).await,
        Command::SkegShards => skeg_shards(shards).await,
        Command::SkegWhoami => skeg_whoami(*tenant, tenant_backend.is_some()),
        Command::SkegAuth { args } => skeg_auth(&args),
        Command::SkegVindexList => skeg_vindex_list(shards, *tenant).await,
        Command::SkegVindexCreate { args } => skeg_vindex_create(&args, shards, *tenant).await,
        Command::SkegVindexDrop { args } => skeg_vindex_drop(&args, shards, *tenant).await,
        Command::SkegVindexConsolidate { args } => {
            skeg_vindex_consolidate(&args, shards, *tenant).await
        }
        Command::SkegVset { args } => skeg_vset(&args, shards, *tenant, tenant_backend).await,
        Command::SkegVmset { args } => skeg_vmset(&args, shards, *tenant, tenant_backend).await,
        Command::SkegVdel { args } => skeg_vdel(&args, shards, *tenant).await,
        Command::SkegQuotaSet { args } => skeg_quota_set(&args, *tenant, tenant_backend),
        Command::SkegQuotaGet { args } => skeg_quota_get(&args, *tenant, tenant_backend),
        Command::SkegQosSet { args } => skeg_qos_set(&args, *tenant, tenant_backend),
        Command::SkegQosGet { args } => skeg_qos_get(&args, *tenant, tenant_backend),
        Command::SkegVsearch { args } => skeg_vsearch(&args, shards, *tenant).await,
        Command::Unknown { name, .. } => unknown_command(&name.to_ascii_uppercase()),
    }
}

/// Dispatcher for command names that did not parse into a typed
/// `Command` variant. After phase 4 every KV / `SKEG.*` verb skeg
/// supports flows through the typed path; this fallback only handles
/// genuinely unknown command names and unknown `SKEG.*` verbs.
fn unknown_command(name: &str) -> Frame {
    Frame::Error(format!("ERR unknown command '{name}'"))
}

/// Parse a bulk-string argument as raw little-endian `f32` bytes.
fn parse_vector(b: &Bytes) -> Result<Vec<f32>, &'static str> {
    if b.len() % 4 != 0 {
        return Err("vector byte length must be a multiple of 4 (f32 LE)");
    }
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn parse_utf8_arg<'a>(b: &'a Bytes, label: &str) -> Result<&'a str, Frame> {
    std::str::from_utf8(b).map_err(|_| Frame::Error(format!("ERR {label} must be UTF-8")))
}

fn parse_u32_arg(b: &Bytes, label: &str) -> Result<u32, Frame> {
    parse_utf8_arg(b, label)?
        .parse()
        .map_err(|_| Frame::Error(format!("ERR {label} must be a non-negative u32")))
}

fn parse_u64_arg(b: &Bytes, label: &str) -> Result<u64, Frame> {
    parse_utf8_arg(b, label)?
        .parse()
        .map_err(|_| Frame::Error(format!("ERR {label} must be a non-negative u64")))
}

fn parse_usize_arg(b: &Bytes, label: &str) -> Result<usize, Frame> {
    parse_utf8_arg(b, label)?
        .parse()
        .map_err(|_| Frame::Error(format!("ERR {label} must be a non-negative integer")))
}

fn parse_kind_arg(b: &Bytes) -> Result<u8, Frame> {
    let s = parse_utf8_arg(b, "kind")?;
    match s.to_ascii_lowercase().as_str() {
        "f32" | "0" => Ok(0),
        "int8" | "1" => Ok(1),
        "binary" | "2" => Ok(2),
        // Disk-tier TurboQuant (sub-int8 RAM on the live write path).
        "tq1" | "3" => Ok(3),
        "tq2" | "4" => Ok(4),
        "tq4" | "5" => Ok(5),
        other => Err(Frame::Error(format!(
            "ERR unknown kind '{other}'; expected f32 | int8 | binary | tq1 | tq2 | tq4"
        ))),
    }
}

fn parse_backend_arg(b: &Bytes) -> Result<u8, Frame> {
    let s = parse_utf8_arg(b, "backend")?;
    match s.to_ascii_lowercase().as_str() {
        "flat" | "0" => Ok(0),
        "disk" | "1" => Ok(1),
        other => Err(Frame::Error(format!(
            "ERR unknown backend '{other}'; expected flat | disk"
        ))),
    }
}

/// `SKEG.VINDEX.CREATE name dim kind backend`. Name is scoped per tenant.
/// Wire byte for tq2, the default tier (recall ~1.0, sub-int8 RAM). Used when
/// `SKEG.VINDEX.CREATE` is called without an explicit kind. Mirrors
/// `parse_kind_arg`'s `"tq2" => 4`.
const DEFAULT_KIND_TQ2: u8 = 4;

async fn skeg_vindex_create(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    // `name dim [kind] backend`: kind is optional and defaults to tq2. Arity (not
    // token shape) disambiguates - kind and backend share numeric aliases (0/1),
    // so a 3-arg call is always [name, dim, backend].
    let (kind_arg, backend_arg) = match args.len() {
        4 => (Some(&args[2]), &args[3]),
        3 => (None, &args[2]),
        _ => {
            return Frame::Error(
                "ERR wrong number of arguments for 'SKEG.VINDEX.CREATE'; want name dim [kind] backend"
                    .into(),
            );
        }
    };
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let dim = match parse_u32_arg(&args[1], "dim") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let kind = match kind_arg {
        Some(b) => match parse_kind_arg(b) {
            Ok(v) => v,
            Err(e) => return e,
        },
        None => DEFAULT_KIND_TQ2,
    };
    let backend = match parse_backend_arg(backend_arg) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.vindex_create(&scoped, dim, kind, backend).await {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VINDEX.DROP name`. Name is scoped per tenant.
/// The tenant id as the `u128` used for vector-quota accounting (`0` for the
/// unscoped default).
fn tenant_u128(tenant: TenantId) -> u128 {
    u128::from_le_bytes(*tenant.as_bytes())
}

async fn skeg_vindex_drop(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'SKEG.VINDEX.DROP'".into());
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.vindex_drop(&scoped, tenant_u128(tenant)).await {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VINDEX.CONSOLIDATE name`. Fold the disk index's streaming delta into
/// its graph (a no-op for flat indices). Useful after a bulk load.
async fn skeg_vindex_consolidate(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'SKEG.VINDEX.CONSOLIDATE'".into());
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.vindex_consolidate(&scoped).await {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VSET name id vector_bytes`. `vector_bytes` is a bulk string
/// carrying raw little-endian `f32` values; its length must be `dim * 4`.
async fn skeg_vset(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    tenant_backend: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.len() != 3 && args.len() != 5 {
        return Frame::Error(
            "ERR wrong number of arguments for 'SKEG.VSET'; want name id vector [PAYLOAD blob]"
                .into(),
        );
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let id = match parse_u64_arg(&args[1], "id") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let vector = match parse_vector(&args[2]) {
        Ok(v) => v,
        Err(e) => return Frame::Error(format!("ERR {e}")),
    };
    // Optional `PAYLOAD <blob>`: an opaque byte buffer stored alongside the
    // vector and returned by a WITHPAYLOAD search.
    let payload = if args.len() == 5 {
        if !args[3].eq_ignore_ascii_case(b"PAYLOAD") {
            return Frame::Error("ERR SKEG.VSET expected PAYLOAD before the blob".into());
        }
        Some(args[4].clone())
    } else {
        None
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // Limit comes from the pluggable backend; `None` (no backend / unlimited)
    // skips quota enforcement entirely.
    let limit = tenant_backend.and_then(|b| b.limits(tenant).max_vectors);
    match shards
        .vset(&scoped, id, vector, tenant_u128(tenant), limit, payload)
        .await
    {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VMSET name (id vector payload)+` - bulk insert. Items fan out
/// concurrently so the durable payload-blob writes batch in the group committer.
/// Returns the number of items inserted.
async fn skeg_vmset(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    tenant_backend: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.len() < 4 || (args.len() - 1) % 3 != 0 {
        return Frame::Error(
            "ERR wrong number of arguments for 'SKEG.VMSET'; want name (id vector payload)+".into(),
        );
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let mut items: Vec<(u64, Vec<f32>, Option<Bytes>)> = Vec::with_capacity((args.len() - 1) / 3);
    let mut i = 1;
    while i < args.len() {
        let id = match parse_u64_arg(&args[i], "id") {
            Ok(v) => v,
            Err(e) => return e,
        };
        let vector = match parse_vector(&args[i + 1]) {
            Ok(v) => v,
            Err(e) => return Frame::Error(format!("ERR {e}")),
        };
        let payload = if args[i + 2].is_empty() {
            None
        } else {
            Some(args[i + 2].clone())
        };
        items.push((id, vector, payload));
        i += 3;
    }
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let limit = tenant_backend.and_then(|b| b.limits(tenant).max_vectors);
    match shards
        .vmset(&scoped, items, tenant_u128(tenant), limit)
        .await
    {
        Ok(n) => Frame::Integer(n as i64),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VDEL name id`.
async fn skeg_vdel(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 2 {
        return Frame::Error("ERR wrong number of arguments for 'SKEG.VDEL'; want name id".into());
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let id = match parse_u64_arg(&args[1], "id") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.vdel(&scoped, id, tenant_u128(tenant)).await {
        Ok(true) => Frame::Integer(1),
        Ok(false) => Frame::Integer(0),
        Err(e) => shard_error(&e),
    }
}

/// Parse a quota limit field: `*` means unlimited (`None`), else a `u64`.
fn parse_quota_limit(b: &Bytes) -> Result<Option<u64>, Frame> {
    if b.as_ref() == b"*" {
        return Ok(None);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(Some)
        .ok_or_else(|| {
            Frame::Error("ERR limit must be a non-negative integer or '*' for unlimited".into())
        })
}

/// Admin-command preamble: require a backend, require the caller be an admin,
/// then resolve the tenant-name arg to a tenant id.
fn admin_target<'a>(
    name_arg: &Bytes,
    caller: TenantId,
    ctx: Option<&'a Arc<dyn TenantBackend>>,
) -> Result<(&'a Arc<dyn TenantBackend>, TenantId), Frame> {
    let Some(backend) = ctx else {
        return Err(Frame::Error(
            "ERR multi-tenant backend not configured".into(),
        ));
    };
    if !backend.is_admin(caller) {
        return Err(Frame::Error("ERR admin privileges required".into()));
    }
    let name = parse_utf8_arg(name_arg, "tenant")?;
    backend
        .resolve_tenant(name)
        .map(|target| (backend, target))
        .ok_or_else(|| Frame::Error("ERR unknown tenant".into()))
}

/// `SKEG.QUOTA.SET tenant max_vectors max_disk_bytes`. Admin only: sets a
/// target tenant's hard quotas. Each limit is a `u64` or `*` (unlimited).
fn skeg_quota_set(args: &[Bytes], caller: TenantId, ctx: Option<&Arc<dyn TenantBackend>>) -> Frame {
    let (backend, target) = match admin_target(&args[0], caller, ctx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let max_vectors = match parse_quota_limit(&args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let max_disk_bytes = match parse_quota_limit(&args[2]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let limits = crate::quota::TenantLimits {
        max_vectors,
        max_disk_bytes,
    };
    match backend.set_limits(target, limits) {
        Ok(()) => Frame::ok(),
        Err(crate::tenant::QuotaAdminError::Unsupported) => {
            Frame::Error("ERR backend does not support setting quotas".into())
        }
    }
}

/// `SKEG.QUOTA.GET tenant`. Admin only: returns `[max_vectors, max_disk_bytes]`
/// as bulk strings, with `*` for an unlimited field.
fn skeg_quota_get(args: &[Bytes], caller: TenantId, ctx: Option<&Arc<dyn TenantBackend>>) -> Frame {
    let (backend, target) = match admin_target(&args[0], caller, ctx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let limits = backend.limits(target);
    let fmt = |o: Option<u64>| o.map_or_else(|| "*".to_string(), |v| v.to_string());
    Frame::Array(vec![
        Frame::Bulk(Bytes::from(fmt(limits.max_vectors))),
        Frame::Bulk(Bytes::from(fmt(limits.max_disk_bytes))),
    ])
}

/// `SKEG.QOS.SET tenant qps burst max_concurrent`. Admin only: sets a target
/// tenant's QoS limits. Each field is a `u32` or `*` (unlimited).
fn skeg_qos_set(args: &[Bytes], caller: TenantId, ctx: Option<&Arc<dyn TenantBackend>>) -> Frame {
    let (backend, target) = match admin_target(&args[0], caller, ctx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let rate = match parse_qos_limit(&args[1]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let burst = match parse_qos_limit(&args[2]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let max_concurrent = match parse_qos_limit(&args[3]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let qos = crate::quota::TenantQos {
        rate,
        burst,
        max_concurrent,
    };
    match backend.set_qos(target, qos) {
        Ok(()) => Frame::ok(),
        Err(crate::tenant::QuotaAdminError::Unsupported) => {
            Frame::Error("ERR backend does not support setting qos".into())
        }
    }
}

/// `SKEG.QOS.GET tenant`. Admin only: returns `[qps, burst, max_concurrent]` as
/// bulk strings, with `*` for an unlimited field.
fn skeg_qos_get(args: &[Bytes], caller: TenantId, ctx: Option<&Arc<dyn TenantBackend>>) -> Frame {
    let (backend, target) = match admin_target(&args[0], caller, ctx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let qos = backend.qos(target);
    let fmt = |o: Option<u32>| o.map_or_else(|| "*".to_string(), |v| v.to_string());
    Frame::Array(vec![
        Frame::Bulk(Bytes::from(fmt(qos.rate))),
        Frame::Bulk(Bytes::from(fmt(qos.burst))),
        Frame::Bulk(Bytes::from(fmt(qos.max_concurrent))),
    ])
}

/// Parse a QoS limit field: `*` = unlimited (`None`), else a `u32`.
fn parse_qos_limit(b: &Bytes) -> Result<Option<u32>, Frame> {
    if b.as_ref() == b"*" {
        return Ok(None);
    }
    std::str::from_utf8(b)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .map(Some)
        .ok_or_else(|| {
            Frame::Error("ERR qos limit must be a non-negative integer or '*' for unlimited".into())
        })
}

/// `SKEG.VSEARCH name k l_search vector_bytes`. Returns an array of
/// `k` pairs `[id (bulk u64-string), score (Double in RESP3 / Bulk in RESP2)]`.
#[tracing::instrument(
    name = "vsearch",
    skip(args, shards),
    fields(
        tenant = %tenant,
        vindex = tracing::field::Empty,
        k = tracing::field::Empty,
        l_search = tracing::field::Empty,
        vector_dim = tracing::field::Empty,
        hits = tracing::field::Empty,
    ),
)]
async fn skeg_vsearch(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if !(4..=7).contains(&args.len()) {
        return Frame::Error(
            "ERR wrong number of arguments for 'SKEG.VSEARCH'; want name k l_search vector \
             [WITHPAYLOAD] [FILTER expr]"
                .into(),
        );
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let k = match parse_usize_arg(&args[1], "k") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let l_search = match parse_u32_arg(&args[2], "l_search") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let query = match parse_vector(&args[3]) {
        Ok(v) => v,
        Err(e) => return Frame::Error(format!("ERR {e}")),
    };
    // Optional tail: `WITHPAYLOAD` and/or `FILTER <expr>`, in either order.
    let mut want_payload = false;
    let mut filter = None;
    let mut i = 4;
    while i < args.len() {
        if args[i].eq_ignore_ascii_case(b"WITHPAYLOAD") {
            want_payload = true;
            i += 1;
        } else if args[i].eq_ignore_ascii_case(b"FILTER") {
            let Some(expr) = args.get(i + 1) else {
                return Frame::Error("ERR SKEG.VSEARCH FILTER needs an expression".into());
            };
            let expr = match parse_utf8_arg(expr, "filter") {
                Ok(s) => s,
                Err(e) => return e,
            };
            match parse_filter(expr) {
                Ok(f) => filter = Some(f),
                Err(e) => return Frame::Error(format!("ERR bad FILTER: {e}")),
            }
            i += 2;
        } else {
            return Frame::Error(format!(
                "ERR unexpected SKEG.VSEARCH argument; want WITHPAYLOAD or FILTER, got '{}'",
                String::from_utf8_lossy(&args[i])
            ));
        }
    }
    let span = tracing::Span::current();
    span.record("vindex", raw_name);
    span.record("k", k);
    span.record("l_search", l_search);
    span.record("vector_dim", query.len());
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards
        .vsearch(
            &scoped,
            query,
            k,
            l_search,
            tenant_u128(tenant),
            want_payload,
            filter,
        )
        .await
    {
        Ok(hits) => {
            span.record("hits", hits.len());
            // Default: flat [id, score, ...] pairs (unchanged). WITHPAYLOAD:
            // [id, score, payload, ...] triples, where payload is a bulk string
            // (empty blobs included) or Null when the id has no stored payload.
            let stride = if want_payload { 3 } else { 2 };
            let mut out = Vec::with_capacity(hits.len() * stride);
            for (id, score, payload) in hits {
                out.push(Frame::Bulk(Bytes::from(id.to_string())));
                out.push(Frame::Double(f64::from(score)));
                if want_payload {
                    out.push(payload.map_or(Frame::Null, Frame::Bulk));
                }
            }
            Frame::Array(out)
        }
        Err(e) => shard_error(&e),
    }
}

/// Enumerate VINDEXes visible to `tenant`. Returns one bulk-string line
/// per VINDEX in `name=<n> dim=<d> kind=<k> backend=<b> n_vectors=<n>`
/// form. Tenant scoping:
///
/// * `TenantId::ZERO` (anonymous or single-tenant): shows only names
///   without a `<hex>::` prefix.
/// * Authenticated tenant: shows only names with the matching
///   `<tenant_hex>::` prefix, with the prefix stripped from the output
///   so the wire form stays the same as single-tenant.
async fn skeg_vindex_list(shards: &ShardSet, tenant: TenantId) -> Frame {
    let prefix = if tenant.is_zero() {
        None
    } else {
        Some(format!("{tenant}::"))
    };
    match shards.vindex_list().await {
        Ok(rows) => {
            let mut body = String::new();
            for (name, dim, kind, backend, n_vectors) in rows {
                let visible_name: &str = match prefix.as_deref() {
                    Some(p) => match name.strip_prefix(p) {
                        Some(rest) => rest,
                        None => continue, // belongs to another tenant
                    },
                    None => {
                        if name.contains("::") {
                            continue; // tenant-scoped name, hidden from ZERO
                        }
                        &name
                    }
                };
                let kind_label = match kind {
                    0 => "f32",
                    1 => "int8",
                    2 => "binary",
                    other => {
                        return Frame::Error(format!(
                            "ERR unexpected kind byte {other} from shard"
                        ));
                    }
                };
                let backend_label = match backend {
                    0 => "flat",
                    1 => "disk",
                    other => {
                        return Frame::Error(format!(
                            "ERR unexpected backend byte {other} from shard"
                        ));
                    }
                };
                body.push_str(&format!(
                    "name={visible_name} dim={dim} kind={kind_label} backend={backend_label} n_vectors={n_vectors}\n",
                ));
            }
            Frame::Bulk(Bytes::from(body))
        }
        Err(e) => shard_error(&e),
    }
}

/// Per-shard stats breakdown. Returns a multi-line bulk string with one
/// row per shard, formatted as `shard=N cache_bytes=X evictions=Y
/// n_keys=Z budget=W`. Easy to parse from redis-cli; the TUI (skeg-top)
/// uses the binary `Op::Shards` for a typed response instead.
async fn skeg_shards(shards: &ShardSet) -> Frame {
    match shards.stats_per_shard().await {
        Ok(rows) => {
            let mut body = String::new();
            for r in rows {
                body.push_str(&format!(
                    "shard={} cache_bytes={} evictions={} n_keys={} budget={}\n",
                    r.shard_id, r.cache_bytes, r.cache_evictions, r.n_keys, r.cache_budget,
                ));
            }
            Frame::Bulk(Bytes::from(body))
        }
        Err(e) => shard_error(&e),
    }
}

/// Report the tenant identity bound to this connection. Useful for
/// drivers that want to assert their AUTH succeeded, and for tests.
fn skeg_whoami(tenant: TenantId, tenancy_enabled: bool) -> Frame {
    let mode = if tenancy_enabled {
        "tenant-aware"
    } else {
        "single-tenant"
    };
    let body = format!("tenant={tenant} mode={mode}");
    Frame::Bulk(Bytes::from(body))
}

/// Placeholder for `SKEG.AUTH <token>` token-based auth. The wire form
/// will pair with `SKEG.AUTH ISSUE` and a per-request token-bearer
/// header in the binary protocol. Reserved here so a client probing the
/// command surface gets a stable error rather than an unknown-command
/// 404.
fn skeg_auth(_args: &[Bytes]) -> Frame {
    Frame::Error("ERR SKEG.AUTH is reserved; use HELLO 3 AUTH user pass for now (v0.2)".into())
}

async fn skeg_stats(shards: &ShardSet) -> Frame {
    match shards.stats().await {
        Ok(s) => {
            // Combine the legacy single-line cache summary with the
            // full Prometheus-flavoured telemetry dump. The first line is
            // kept verbatim so existing redis-cli scripts that grep for
            // `cache_bytes=` keep working; the rest is the telemetry
            // section, separated by a blank line.
            let cache_line = format!(
                "cache_bytes={} evictions={} n_keys={} budget={}",
                s.cache_bytes, s.cache_evictions, s.n_keys, s.cache_budget,
            );
            let body = format!("{cache_line}\n\n{}", skeg_telemetry::stats::dump_text());
            Frame::Bulk(Bytes::from(body))
        }
        Err(e) => shard_error(&e),
    }
}

fn shard_error(e: &crate::shard::ShardError) -> Frame {
    warn!("shard error: {e}");
    Frame::Error(format!("ERR {e}"))
}

/// `INCRBY` / `DECRBY` body after the parser has unpacked the delta
/// and the dispatcher has folded the sign for `DECRBY`. The legacy
/// `kv_incrby_arg` path that re-parsed the integer is gone now that
/// `skeg-resp3` carries the typed `delta: i64`.
async fn kv_incrby_apply(
    key: &Bytes,
    delta: i64,
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if anon_key_collides_with_tenant(tenant, key, ctx) {
        return anon_forgery_error();
    }
    let k = scope_key(tenant, key);
    incr_apply(k.as_bytes(), delta, shards, k.accounting_tenant()).await
}

/// `SELECT db`. Skeg has one logical DB; only index 0 succeeds.
fn kv_select_db(db: i64) -> Frame {
    if db == 0 {
        Frame::ok()
    } else {
        Frame::Error("ERR DB index out of range (skeg only supports DB 0)".into())
    }
}

async fn kv_get(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'GET'".into());
    }
    if anon_key_collides_with_tenant(tenant, &args[0], ctx) {
        return anon_forgery_error();
    }
    let k = scope_key(tenant, &args[0]);
    match shards.tenant(k.accounting_tenant()).get(k.as_bytes()).await {
        Ok(Some(v)) => Frame::Bulk(v),
        Ok(None) => Frame::Null,
        Err(e) => shard_error(&e),
    }
}

async fn kv_set(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.len() != 2 {
        return Frame::Error("ERR wrong number of arguments for 'SET'".into());
    }
    if anon_key_collides_with_tenant(tenant, &args[0], ctx) {
        return anon_forgery_error();
    }
    let k = scope_key(tenant, &args[0]);
    // Disk quota from the pluggable backend; `None` skips enforcement.
    let disk_limit = ctx.and_then(|b| b.limits(tenant).max_disk_bytes);
    match shards
        .tenant(k.accounting_tenant())
        .with_disk_limit(disk_limit)
        .set(k.as_bytes(), &args[1], DEFAULT_DURABILITY)
        .await
    {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

async fn kv_del(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.is_empty() {
        return Frame::Error("ERR wrong number of arguments for 'DEL'".into());
    }
    for key in args {
        if anon_key_collides_with_tenant(tenant, key, ctx) {
            return anon_forgery_error();
        }
    }
    let mut count: i64 = 0;
    for key in args {
        let k = scope_key(tenant, key);
        match shards.del(k.as_bytes(), DEFAULT_DURABILITY).await {
            Ok(true) => count += 1,
            Ok(false) => {}
            Err(e) => return shard_error(&e),
        }
    }
    Frame::Integer(count)
}

async fn kv_exists(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.is_empty() {
        return Frame::Error("ERR wrong number of arguments for 'EXISTS'".into());
    }
    for key in args {
        if anon_key_collides_with_tenant(tenant, key, ctx) {
            return anon_forgery_error();
        }
    }
    let mut count: i64 = 0;
    for key in args {
        let k = scope_key(tenant, key);
        match shards.tenant(k.accounting_tenant()).get(k.as_bytes()).await {
            Ok(Some(_)) => count += 1,
            Ok(None) => {}
            Err(e) => return shard_error(&e),
        }
    }
    Frame::Integer(count)
}

async fn kv_mget(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.is_empty() {
        return Frame::Error("ERR wrong number of arguments for 'MGET'".into());
    }
    for key in args {
        if anon_key_collides_with_tenant(tenant, key, ctx) {
            return anon_forgery_error();
        }
    }
    let mut out = Vec::with_capacity(args.len());
    for key in args {
        let k = scope_key(tenant, key);
        match shards.tenant(k.accounting_tenant()).get(k.as_bytes()).await {
            Ok(Some(v)) => out.push(Frame::Bulk(v)),
            Ok(None) => out.push(Frame::Null),
            Err(e) => return shard_error(&e),
        }
    }
    Frame::Array(out)
}

async fn kv_mset(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.is_empty() || args.len() % 2 != 0 {
        return Frame::Error("ERR wrong number of arguments for 'MSET'".into());
    }
    for chunk in args.chunks(2) {
        if anon_key_collides_with_tenant(tenant, &chunk[0], ctx) {
            return anon_forgery_error();
        }
    }
    for chunk in args.chunks(2) {
        let k = scope_key(tenant, &chunk[0]);
        match shards
            .set(k.as_bytes(), &chunk[1], DEFAULT_DURABILITY)
            .await
        {
            Ok(()) => {}
            Err(e) => return shard_error(&e),
        }
    }
    Frame::ok()
}

async fn kv_incr_by(
    args: &[Bytes],
    shards: &ShardSet,
    sign: i64,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'INCR/DECR'".into());
    }
    if anon_key_collides_with_tenant(tenant, &args[0], ctx) {
        return anon_forgery_error();
    }
    let k = scope_key(tenant, &args[0]);
    incr_apply(k.as_bytes(), sign, shards, k.accounting_tenant()).await
}

async fn incr_apply(key: &Bytes, delta: i64, shards: &ShardSet, tenant: u128) -> Frame {
    let store = shards.tenant(tenant);
    let current: i64 = match store.get(key).await {
        Ok(Some(b)) => match std::str::from_utf8(&b).ok().and_then(|s| s.parse().ok()) {
            Some(n) => n,
            None => return Frame::Error("ERR value is not an integer or out of range".into()),
        },
        Ok(None) => 0,
        Err(e) => return shard_error(&e),
    };
    let new = match current.checked_add(delta) {
        Some(v) => v,
        None => return Frame::Error("ERR increment or decrement would overflow".into()),
    };
    let body = Bytes::from(new.to_string());
    match store.set(key, &body, DEFAULT_DURABILITY).await {
        Ok(()) => Frame::Integer(new),
        Err(e) => shard_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, is_pipelineable};
    use bytes::Bytes;

    /// Safety guard: order-dependent commands must NEVER be pipelined
    /// (concurrent dispatch would reorder their effects on a shared key -
    /// lost INCR updates, non-deterministic SET, stale read-after-write).
    /// The vector write path is keyed by distinct ids (upsert semantics) and
    /// VSEARCH is read-only, so those stay concurrent.
    #[test]
    fn only_order_independent_commands_are_pipelineable() {
        let k = || Bytes::from_static(b"k");
        // Must be barriers (serial): every scalar KV verb.
        for cmd in [
            Command::Get { key: k() },
            Command::Set {
                key: k(),
                value: k(),
            },
            Command::Del { keys: vec![k()] },
            Command::Exists { keys: vec![k()] },
            Command::Mget { keys: vec![k()] },
            Command::Mset {
                pairs: vec![(k(), k())],
            },
            Command::Incr { key: k() },
            Command::Decr { key: k() },
        ] {
            assert!(!is_pipelineable(&cmd), "{cmd:?} must be a serial barrier");
        }
        // Safe to pipeline.
        for cmd in [
            Command::SkegVset { args: vec![k()] },
            Command::SkegVmset { args: vec![k()] },
            Command::SkegVsearch { args: vec![k()] },
            Command::SkegVdel { args: vec![k()] },
            Command::Ping(None),
            Command::Echo(k()),
        ] {
            assert!(is_pipelineable(&cmd), "{cmd:?} should pipeline");
        }
    }

    /// Local deterministic tenant id from a string. Mirrors what the
    /// real tenant backend does (xxh3_128 of the name) for tests that
    /// need stable ids without pulling in a full backend impl.
    fn tid_from_name(name: &str) -> TenantId {
        let h = xxhash_rust::xxh3::xxh3_128(name.as_bytes());
        TenantId::from_bytes(h.to_le_bytes())
    }

    use super::*;
    use tempfile::TempDir;

    #[test]
    fn anon_cannot_forge_tenant_scope_via_double_colon() {
        // The whole cross-tenant guard: an anonymous (ZERO) connection must not
        // be able to smuggle another tenant's `<hex>::` prefix into a name.
        let victim = tid_from_name("victim");
        let forged = format!("{victim}::secret"); // what an attacker would type
        assert!(
            scope_vindex_or_reject(TenantId::ZERO, &forged).is_err(),
            "anon must be rejected when the name contains '::'"
        );
        // A plain anon name is accepted unchanged...
        assert_eq!(
            scope_vindex_or_reject(TenantId::ZERO, "idx").unwrap(),
            "idx"
        );
        // ...and an authenticated tenant's own name gets its real prefix, which
        // can never collide with an accepted anon name (those have no '::').
        assert_eq!(
            scope_vindex_or_reject(victim, "idx").unwrap(),
            format!("{victim}::idx")
        );
        // A tenant also cannot inject a second scope.
        assert!(scope_vindex_or_reject(victim, "a::b").is_err());
    }

    fn args(parts: &[&str]) -> Vec<Bytes> {
        parts
            .iter()
            .map(|s| Bytes::copy_from_slice(s.as_bytes()))
            .collect()
    }

    #[test]
    fn command_cost_vsearch_sums_k_and_l_search() {
        // SKEG.VSEARCH idx k=10 l_search=128 vec -> 138 credits.
        let cmd = Command::SkegVsearch {
            args: args(&["idx", "10", "128", "vec"]),
        };
        assert_eq!(command_cost(&cmd), 138);
    }

    #[test]
    fn command_cost_kv_is_one() {
        let cmd = Command::Get {
            key: Bytes::from_static(b"k"),
        };
        assert_eq!(command_cost(&cmd), 1);
    }

    #[test]
    fn command_kind_classifies_for_rbac() {
        let cases = [
            (
                Command::Get {
                    key: Bytes::from_static(b"k"),
                },
                CommandKind::KvRead,
            ),
            (
                Command::Set {
                    key: Bytes::from_static(b"k"),
                    value: Bytes::from_static(b"v"),
                },
                CommandKind::KvWrite,
            ),
            (
                Command::SkegVsearch {
                    args: args(&["i", "1", "1", "v"]),
                },
                CommandKind::VectorRead,
            ),
            (
                Command::SkegVset {
                    args: args(&["i", "1", "v"]),
                },
                CommandKind::VectorWrite,
            ),
            (
                Command::SkegVindexCreate {
                    args: args(&["i", "4", "1", "1"]),
                },
                CommandKind::VindexCreate,
            ),
            (
                Command::SkegVindexDrop { args: args(&["i"]) },
                CommandKind::VindexDrop,
            ),
            (
                Command::SkegQosSet { args: args(&["t"]) },
                CommandKind::Admin,
            ),
            (Command::Ping(None), CommandKind::Meta),
        ];
        for (cmd, want) in cases {
            assert_eq!(command_kind(&cmd), want, "misclassified {cmd:?}");
        }
    }

    /// A backend that records every `Admission.op` it sees and refuses
    /// `VindexDrop` - the minimal per-command RBAC the new seam enables.
    struct RbacBackend {
        seen: std::sync::Mutex<Vec<CommandKind>>,
    }

    impl TenantBackend for RbacBackend {
        fn verify_login(&self, _user: &str, _password: &[u8]) -> Option<TenantId> {
            None
        }
        fn has_tenant(&self, _id: TenantId) -> bool {
            false
        }
        fn admit(&self, a: Admission) -> Result<crate::AdmitGuard, crate::AdmitRejected> {
            self.seen.lock().unwrap().push(a.op);
            if a.op == CommandKind::VindexDrop {
                return Err(crate::AdmitRejected {
                    message: "FORBIDDEN drop denied".into(),
                });
            }
            Ok(crate::AdmitGuard::allow())
        }
    }

    #[tokio::test]
    async fn admit_refuses_command_by_kind() {
        let (_dir, shards) = fresh_shards().await;
        let concrete = std::sync::Arc::new(RbacBackend {
            seen: std::sync::Mutex::new(Vec::new()),
        });
        let backend: std::sync::Arc<dyn TenantBackend> = concrete.clone();
        let mut state = ConnectionState::new(0);
        let mut tenant = tid_from_name("t");

        // VINDEX.DROP is refused at the gate, by op, before touching the shard.
        let f = dispatch_command(
            Command::SkegVindexDrop {
                args: args(&["idx"]),
            },
            &mut state,
            &mut tenant,
            &shards,
            Some(&backend),
        )
        .await;
        assert!(
            matches!(&f, Frame::Error(e) if e.contains("FORBIDDEN")),
            "VINDEX.DROP must be refused by the gate, got {f:?}"
        );

        // GET passes the gate (it may then fail on a missing key, but never with
        // the admit rejection).
        let f = dispatch_command(
            Command::Get {
                key: Bytes::from_static(b"k"),
            },
            &mut state,
            &mut tenant,
            &shards,
            Some(&backend),
        )
        .await;
        assert!(
            !matches!(&f, Frame::Error(e) if e.contains("FORBIDDEN")),
            "GET must pass the gate, got {f:?}"
        );

        let seen = concrete.seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![CommandKind::VindexDrop, CommandKind::KvRead],
            "admit must see each command's classified op"
        );
    }

    #[tokio::test]
    async fn vindex_create_defaults_to_tq2_tier() {
        // A disk VINDEX.CREATE with the kind omitted (3 args) must default to a
        // sub-int8 tier (tq2). Verified behaviorally: same data, the default
        // index is leaner in RAM than an explicit int8 one.
        let (_dir, shards) = fresh_shards().await;
        let mut state = ConnectionState::new(0);
        let mut tenant = TenantId::ZERO;
        let dim = 64usize;
        let n = 2000u64;

        for (name, create_args) in [
            ("def", args(&["def", "64", "disk"])), // 3 args -> kind defaults to tq2
            ("i8", args(&["i8", "64", "int8", "disk"])), // 4 args -> explicit int8
        ] {
            let f = dispatch_command(
                Command::SkegVindexCreate { args: create_args },
                &mut state,
                &mut tenant,
                &shards,
                None,
            )
            .await;
            assert!(!matches!(f, Frame::Error(_)), "create {name} failed: {f:?}");
            for id in 1..=n {
                let mut v = vec![0f32; dim];
                v[0] = id as f32;
                v[1 + (id as usize % (dim - 1))] = 1.0;
                shards.vset(name, id, v, 0, None, None).await.unwrap();
            }
            shards.vindex_consolidate(name).await.unwrap();
        }

        let stats = shards.control_handle().open_indices().await;
        let rb = |idx: &str| {
            stats
                .iter()
                .filter(|s| s.index == idx)
                .map(|s| s.resident_bytes)
                .sum::<usize>()
        };
        assert!(
            rb("def") < rb("i8"),
            "default tier must be leaner than int8: def={} i8={}",
            rb("def"),
            rb("i8")
        );
    }

    #[tokio::test]
    async fn vindex_create_tq1_bad_dim_errors_not_panics() {
        let (_dir, shards) = fresh_shards().await;
        let t = TenantId::ZERO;
        // 100 % 8 != 0: tq1 cannot pack it. Must be a clean error, and reaching
        // the next line at all proves the server did not panic.
        let bad = skeg_vindex_create(&args(&["bad", "100", "tq1", "disk"]), &shards, t).await;
        assert!(matches!(bad, Frame::Error(_)));
        // An 8-aligned dim is accepted.
        let ok = skeg_vindex_create(&args(&["good", "128", "tq1", "disk"]), &shards, t).await;
        assert!(matches!(ok, Frame::Simple(ref s) if s == "OK"));
    }

    async fn fresh_shards() -> (TempDir, ShardSet) {
        let dir = TempDir::new().unwrap();
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        (dir, shards)
    }

    #[tokio::test]
    async fn mget_returns_value_per_key_in_order() {
        let (_dir, shards) = fresh_shards().await;
        // Seed two keys; leave the third missing so MGET must produce a
        // Null in that slot while preserving order.
        let _ = kv_set(&args(&["k1", "v1"]), &shards, TenantId::ZERO, None).await;
        let _ = kv_set(&args(&["k3", "v3"]), &shards, TenantId::ZERO, None).await;

        let resp = kv_mget(&args(&["k1", "k2", "k3"]), &shards, TenantId::ZERO, None).await;
        match resp {
            Frame::Array(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(items[0], Frame::Bulk(ref b) if &b[..] == b"v1"));
                assert!(matches!(items[1], Frame::Null));
                assert!(matches!(items[2], Frame::Bulk(ref b) if &b[..] == b"v3"));
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mset_then_get_each_key() {
        let (_dir, shards) = fresh_shards().await;
        let resp = kv_mset(
            &args(&["a", "1", "b", "2", "c", "3"]),
            &shards,
            TenantId::ZERO,
            None,
        )
        .await;
        assert!(matches!(resp, Frame::Simple(ref s) if s == "OK"));
        for (k, v) in [("a", "1"), ("b", "2"), ("c", "3")] {
            let r = kv_get(&args(&[k]), &shards, TenantId::ZERO, None).await;
            assert!(matches!(r, Frame::Bulk(ref b) if &b[..] == v.as_bytes()));
        }
    }

    #[tokio::test]
    async fn mset_rejects_odd_arity() {
        let (_dir, shards) = fresh_shards().await;
        let resp = kv_mset(&args(&["k1", "v1", "k2"]), &shards, TenantId::ZERO, None).await;
        assert!(matches!(resp, Frame::Error(ref e) if e.contains("wrong number")));
    }

    // Raw f32-LE bytes, the on-wire vector encoding parse_vector expects.
    fn vec_arg(v: &[f32]) -> Bytes {
        Bytes::from(v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())
    }

    // End-to-end through the RESP3 handlers: VSET ... PAYLOAD then
    // VSEARCH ... WITHPAYLOAD returns the blob (the surface is callable, not
    // just documented). Also covers the keyword-typo rejection.
    #[tokio::test]
    async fn vset_payload_then_vsearch_withpayload() {
        let (_dir, shards) = fresh_shards().await;
        let t = TenantId::ZERO;
        let q = vec_arg(&[1.0, 0.0]);

        let created = skeg_vindex_create(&args(&["idx", "2", "f32", "flat"]), &shards, t).await;
        assert!(matches!(created, Frame::Simple(ref s) if s == "OK"));

        let set = skeg_vset(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"1"),
                q.clone(),
                Bytes::from_static(b"PAYLOAD"),
                Bytes::from_static(b"hello"),
            ],
            &shards,
            t,
            None,
        )
        .await;
        assert!(matches!(set, Frame::Simple(ref s) if s == "OK"));

        let resp = skeg_vsearch(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"5"),
                Bytes::from_static(b"0"),
                q.clone(),
                Bytes::from_static(b"WITHPAYLOAD"),
            ],
            &shards,
            t,
        )
        .await;
        match resp {
            // One hit, encoded as an [id, score, payload] triple.
            Frame::Array(items) => {
                assert_eq!(items.len(), 3);
                assert!(matches!(items[0], Frame::Bulk(ref b) if &b[..] == b"1"));
                assert!(matches!(items[1], Frame::Double(_)));
                assert!(matches!(items[2], Frame::Bulk(ref b) if &b[..] == b"hello"));
            }
            other => panic!("expected Array triple, got {other:?}"),
        }

        // Without WITHPAYLOAD: flat [id, score] pair, no payload slot.
        let plain = skeg_vsearch(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"5"),
                Bytes::from_static(b"0"),
                q.clone(),
            ],
            &shards,
            t,
        )
        .await;
        assert!(matches!(plain, Frame::Array(ref items) if items.len() == 2));

        // A mistyped trailing keyword is rejected, not silently ignored.
        let bad = skeg_vsearch(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"5"),
                Bytes::from_static(b"0"),
                q,
                Bytes::from_static(b"NOPE"),
            ],
            &shards,
            t,
        )
        .await;
        assert!(matches!(bad, Frame::Error(_)));
    }

    // End-to-end through the RESP3 handler: VSET ... PAYLOAD then
    // VSEARCH ... FILTER returns only the matching ids (surface is callable).
    #[tokio::test]
    async fn vsearch_filter_selects_matching_ids() {
        let (_dir, shards) = fresh_shards().await;
        let t = TenantId::ZERO;
        let created = skeg_vindex_create(&args(&["idx", "2", "f32", "flat"]), &shards, t).await;
        assert!(matches!(created, Frame::Simple(ref s) if s == "OK"));

        for (id, who) in [("1", "user=bob"), ("2", "user=alice"), ("3", "user=alice")] {
            let set = skeg_vset(
                &[
                    Bytes::from_static(b"idx"),
                    Bytes::copy_from_slice(id.as_bytes()),
                    vec_arg(&[1.0, 0.0]),
                    Bytes::from_static(b"PAYLOAD"),
                    Bytes::copy_from_slice(who.as_bytes()),
                ],
                &shards,
                t,
                None,
            )
            .await;
            assert!(matches!(set, Frame::Simple(ref s) if s == "OK"));
        }

        let resp = skeg_vsearch(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"10"),
                Bytes::from_static(b"0"),
                vec_arg(&[1.0, 0.0]),
                Bytes::from_static(b"FILTER"),
                Bytes::from_static(b"user = alice"),
            ],
            &shards,
            t,
        )
        .await;
        match resp {
            // Two alice hits as [id, score] pairs; bob's id 1 excluded.
            Frame::Array(items) => {
                assert_eq!(items.len(), 4);
                let ids: Vec<&[u8]> = items
                    .iter()
                    .step_by(2)
                    .map(|f| match f {
                        Frame::Bulk(b) => &b[..],
                        other => panic!("expected id bulk, got {other:?}"),
                    })
                    .collect();
                assert!(ids.contains(&&b"2"[..]) && ids.contains(&&b"3"[..]));
                assert!(!ids.contains(&&b"1"[..]), "bob's id 1 must be filtered out");
            }
            other => panic!("expected Array, got {other:?}"),
        }

        // A malformed filter is a clean error, not a panic.
        let bad = skeg_vsearch(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"10"),
                Bytes::from_static(b"0"),
                vec_arg(&[1.0, 0.0]),
                Bytes::from_static(b"FILTER"),
                Bytes::from_static(b"user =="),
            ],
            &shards,
            t,
        )
        .await;
        assert!(matches!(bad, Frame::Error(ref e) if e.contains("bad FILTER")));
    }

    #[tokio::test]
    async fn incr_starts_at_one_for_missing_key() {
        let (_dir, shards) = fresh_shards().await;
        let resp = kv_incr_by(&args(&["counter"]), &shards, 1, TenantId::ZERO, None).await;
        assert!(matches!(resp, Frame::Integer(1)));
        // Stored as a UTF-8 integer, GET-readable.
        let g = kv_get(&args(&["counter"]), &shards, TenantId::ZERO, None).await;
        assert!(matches!(g, Frame::Bulk(ref b) if &b[..] == b"1"));
    }

    #[tokio::test]
    async fn incr_decr_round_trip() {
        let (_dir, shards) = fresh_shards().await;
        for _ in 0..5 {
            let _ = kv_incr_by(&args(&["c"]), &shards, 1, TenantId::ZERO, None).await;
        }
        let r = kv_incr_by(&args(&["c"]), &shards, -1, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Integer(4)));
    }

    #[tokio::test]
    async fn incrby_applies_signed_delta() {
        let (_dir, shards) = fresh_shards().await;
        let key = Bytes::from_static(b"c");
        let r = kv_incrby_apply(&key, 42, &shards, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Integer(42)));
        let r = kv_incrby_apply(&key, -10, &shards, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Integer(32)));
    }

    #[tokio::test]
    async fn incr_rejects_non_integer_value() {
        let (_dir, shards) = fresh_shards().await;
        let _ = kv_set(
            &args(&["bad", "not-a-number"]),
            &shards,
            TenantId::ZERO,
            None,
        )
        .await;
        let r = kv_incr_by(&args(&["bad"]), &shards, 1, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Error(ref e) if e.contains("not an integer")));
    }

    #[tokio::test]
    async fn incr_rejects_overflow() {
        let (_dir, shards) = fresh_shards().await;
        let max = i64::MAX.to_string();
        let _ = kv_set(&args(&["big", &max]), &shards, TenantId::ZERO, None).await;
        let r = kv_incr_by(&args(&["big"]), &shards, 1, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Error(ref e) if e.contains("overflow")));
    }

    #[tokio::test]
    async fn skeg_stats_returns_a_bulk_summary() {
        let (_dir, shards) = fresh_shards().await;
        let resp = skeg_stats(&shards).await;
        match resp {
            Frame::Bulk(b) => {
                let s = std::str::from_utf8(&b).unwrap();
                assert!(s.contains("cache_bytes="));
                assert!(s.contains("evictions="));
                assert!(s.contains("n_keys="));
                assert!(s.contains("budget="));
            }
            other => panic!("expected Bulk, got {other:?}"),
        }
    }

    #[test]
    fn skeg_namespace_rejects_unknown_verb() {
        // Unknown `SKEG.*` verbs come through the parser as `Unknown` and the
        // dispatcher emits the legacy `ERR unknown command 'SKEG.<verb>'`.
        let resp = unknown_command("SKEG.WHATEVER");
        assert!(matches!(resp, Frame::Error(ref e) if e.contains("'SKEG.WHATEVER'")));
    }

    #[tokio::test]
    async fn skeg_whoami_reports_zero_when_anonymous() {
        // No tenant context wired in: WHOAMI must say single-tenant + ZERO.
        let f = skeg_whoami(TenantId::ZERO, false);
        match f {
            Frame::Bulk(b) => {
                let s = std::str::from_utf8(&b).unwrap();
                assert!(s.contains("tenant=00000000000000000000000000000000"));
                assert!(s.contains("mode=single-tenant"));
            }
            other => panic!("expected Bulk, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn skeg_whoami_reports_resolved_tenant() {
        let alice = tid_from_name("alice");
        let f = skeg_whoami(alice, true);
        match f {
            Frame::Bulk(b) => {
                let s = std::str::from_utf8(&b).unwrap();
                assert!(s.contains(&format!("tenant={alice}")));
                assert!(s.contains("mode=tenant-aware"));
            }
            other => panic!("expected Bulk, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn scoped_keys_isolate_two_tenants() {
        // Same logical key "k" written by alice and bob must not collide
        // on disk: scope_key prefixes with the tenant id, so each lands
        // in a distinct entry in the shard.
        let (_dir, shards) = fresh_shards().await;
        let alice = tid_from_name("alice");
        let bob = tid_from_name("bob");

        let _ = kv_set(&args(&["k", "alice-value"]), &shards, alice, None).await;
        let _ = kv_set(&args(&["k", "bob-value"]), &shards, bob, None).await;

        let r_alice = kv_get(&args(&["k"]), &shards, alice, None).await;
        assert!(matches!(r_alice, Frame::Bulk(ref b) if &b[..] == b"alice-value"));
        let r_bob = kv_get(&args(&["k"]), &shards, bob, None).await;
        assert!(matches!(r_bob, Frame::Bulk(ref b) if &b[..] == b"bob-value"));

        // And the anonymous (ZERO) view must not see either: ZERO writes
        // are unprefixed and the scoped writes carry a non-zero prefix.
        let r_anon = kv_get(&args(&["k"]), &shards, TenantId::ZERO, None).await;
        assert!(matches!(r_anon, Frame::Null));
    }

    #[tokio::test]
    async fn tenant_del_only_affects_own_namespace() {
        let (_dir, shards) = fresh_shards().await;
        let alice = tid_from_name("alice");
        let bob = tid_from_name("bob");
        let _ = kv_set(&args(&["k", "av"]), &shards, alice, None).await;
        let _ = kv_set(&args(&["k", "bv"]), &shards, bob, None).await;

        // Alice deletes her own "k"; Bob's "k" must survive.
        let d = kv_del(&args(&["k"]), &shards, alice, None).await;
        assert!(matches!(d, Frame::Integer(1)));
        let r_bob = kv_get(&args(&["k"]), &shards, bob, None).await;
        assert!(matches!(r_bob, Frame::Bulk(ref b) if &b[..] == b"bv"));
        let r_alice = kv_get(&args(&["k"]), &shards, alice, None).await;
        assert!(matches!(r_alice, Frame::Null));
    }

    #[tokio::test]
    async fn tenant_incr_counters_are_isolated() {
        // Two tenants both incrementing the same logical key: counters
        // must advance independently.
        let (_dir, shards) = fresh_shards().await;
        let alice = tid_from_name("alice");
        let bob = tid_from_name("bob");

        for _ in 0..3 {
            let _ = kv_incr_by(&args(&["hits"]), &shards, 1, alice, None).await;
        }
        for _ in 0..5 {
            let _ = kv_incr_by(&args(&["hits"]), &shards, 1, bob, None).await;
        }
        let r = kv_get(&args(&["hits"]), &shards, alice, None).await;
        assert!(matches!(r, Frame::Bulk(ref b) if &b[..] == b"3"));
        let r = kv_get(&args(&["hits"]), &shards, bob, None).await;
        assert!(matches!(r, Frame::Bulk(ref b) if &b[..] == b"5"));
    }
}
