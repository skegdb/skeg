//! RESP3 connection handler.
//!
//! Per-connection task that reads bytes from a TCP socket into a `FrameDecoder`,
//! parses them into `Command`s, dispatches to the `ShardSet`, encodes the
//! response back via `encode_frame`, and writes to the socket.
//!
//! Mirrors the binary-protocol `handler.rs` but speaks Redis wire (RESP2/RESP3).
//! New connections default to RESP2 until `HELLO 3` upgrades them.
//!
//! Wire commands supported in this iteration (M9 v0.1 KV subset):
//! - `HELLO [version [AUTH user pass] [SETNAME name]]` - protocol negotiation.
//! - `PING [msg]` / `ECHO msg` - protocol-only.
//! - `GET key` / `SET key value` / `DEL key [key ...]` / `EXISTS key [key ...]`.
//! - `SELECT 0` accepted as no-op (driver compat), `SELECT N>0` rejected.
//!
//! Out of scope here (later v0.1 / v0.2): SET options (EX/PX/NX/XX), EXPIRE/TTL,
//! INFO/STATS/DBSIZE/COMMAND, SHUTDOWN, vector ops, async maintenance, AUTH model.

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

use crate::shard::ShardSet;
use crate::tenant::{AnonymousPolicy, TenantBackend, TenantId};

/// Format a VINDEX name with the tenant scope. `TenantId::ZERO` returns the
/// raw name (single-tenant deployments stay byte-identical to pre-tenancy).
fn scoped_vindex_name(tenant: TenantId, name: &str) -> String {
    if tenant.is_zero() {
        name.to_string()
    } else {
        format!("{tenant}::{name}")
    }
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

    loop {
        let frame = match decoder.decode() {
            Ok(Some(f)) => f,
            Ok(None) => match stream.read_buf(decoder.buf_mut()).await {
                Ok(0) => break,
                Ok(_) => continue,
                Err(e) => {
                    warn!(?peer, "read error: {e}");
                    break;
                }
            },
            Err(e) => {
                warn!(?peer, "RESP3 parse error: {e}");
                let err = Frame::Error(format!("ERR protocol: {e}"));
                out.clear();
                encode_frame(&err, state.version, &mut out);
                let _ = stream.write_all(&out).await;
                break;
            }
        };

        let response = match parse_command(frame) {
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
            break;
        }
    }

    debug!(?peer, "RESP3 connection closed");
}

async fn dispatch_command(
    cmd: Command,
    state: &mut ConnectionState,
    tenant: &mut TenantId,
    shards: &ShardSet,
    tenant_backend: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
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
        Command::Set { key, value } => {
            kv_set(&[key, value], shards, *tenant, tenant_backend).await
        }
        Command::Del { keys } => kv_del(&keys, shards, *tenant, tenant_backend).await,
        Command::Exists { keys } => kv_exists(&keys, shards, *tenant, tenant_backend).await,
        Command::Mget { keys } => kv_mget(&keys, shards, *tenant, tenant_backend).await,
        Command::Mset { pairs } => {
            let args: Vec<Bytes> = pairs.into_iter().flat_map(|(k, v)| [k, v]).collect();
            kv_mset(&args, shards, *tenant, tenant_backend).await
        }
        Command::Incr { key } => {
            kv_incr_by(std::slice::from_ref(&key), shards, 1, *tenant, tenant_backend).await
        }
        Command::Decr { key } => {
            kv_incr_by(std::slice::from_ref(&key), shards, -1, *tenant, tenant_backend).await
        }
        Command::IncrBy { key, delta } => kv_incrby_apply(&key, delta, shards, *tenant, tenant_backend).await,
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
        Command::SkegVset { args } => skeg_vset(&args, shards, *tenant).await,
        Command::SkegVdel { args } => skeg_vdel(&args, shards, *tenant).await,
        Command::SkegVsearch { args } => skeg_vsearch(&args, shards, *tenant).await,
        Command::Unknown { name, args } => {
            dispatch_unknown(
                &name.to_ascii_uppercase(),
                args,
                shards,
                *tenant,
                tenant_backend,
            )
            .await
        }
    }
}

/// Dispatcher for command names that did not parse into a typed
/// `Command` variant. After phase 4 every KV / `SKEG.*` verb skeg
/// supports flows through the typed path; this fallback only handles
/// genuinely unknown command names and unknown `SKEG.*` verbs.
async fn dispatch_unknown(
    name: &str,
    _args: Vec<Bytes>,
    _shards: &ShardSet,
    _tenant: TenantId,
    _tenant_backend: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
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
        other => Err(Frame::Error(format!(
            "ERR unknown kind '{other}'; expected f32 | int8 | binary"
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
async fn skeg_vindex_create(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 4 {
        return Frame::Error(
            "ERR wrong number of arguments for 'SKEG.VINDEX.CREATE'; want name dim kind backend"
                .into(),
        );
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    if raw_name.contains("::") {
        return Frame::Error(
            "ERR VINDEX name must not contain '::' (reserved for tenant scoping)".into(),
        );
    }
    let dim = match parse_u32_arg(&args[1], "dim") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let kind = match parse_kind_arg(&args[2]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let backend = match parse_backend_arg(&args[3]) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let scoped = scoped_vindex_name(tenant, raw_name);
    match shards.vindex_create(&scoped, dim, kind, backend).await {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VINDEX.DROP name`. Name is scoped per tenant.
async fn skeg_vindex_drop(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'SKEG.VINDEX.DROP'".into());
    }
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = scoped_vindex_name(tenant, raw_name);
    match shards.vindex_drop(&scoped).await {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VSET name id vector_bytes`. `vector_bytes` is a bulk string
/// carrying raw little-endian `f32` values; its length must be `dim * 4`.
async fn skeg_vset(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 3 {
        return Frame::Error(
            "ERR wrong number of arguments for 'SKEG.VSET'; want name id vector".into(),
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
    let scoped = scoped_vindex_name(tenant, raw_name);
    match shards.vset(&scoped, id, vector).await {
        Ok(()) => Frame::ok(),
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
    let scoped = scoped_vindex_name(tenant, raw_name);
    match shards.vdel(&scoped, id).await {
        Ok(true) => Frame::Integer(1),
        Ok(false) => Frame::Integer(0),
        Err(e) => shard_error(&e),
    }
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
    if args.len() != 4 {
        return Frame::Error(
            "ERR wrong number of arguments for 'SKEG.VSEARCH'; want name k l_search vector".into(),
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
    let span = tracing::Span::current();
    span.record("vindex", raw_name);
    span.record("k", k);
    span.record("l_search", l_search);
    span.record("vector_dim", query.len());
    let scoped = scoped_vindex_name(tenant, raw_name);
    match shards.vsearch(&scoped, query, k, l_search).await {
        Ok(hits) => {
            span.record("hits", hits.len());
            let mut out = Vec::with_capacity(hits.len() * 2);
            for (id, score) in hits {
                out.push(Frame::Bulk(Bytes::from(id.to_string())));
                out.push(Frame::Double(f64::from(score)));
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
    incr_apply(k.as_bytes(), delta, shards).await
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
    match shards.get(k.as_bytes()).await {
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
    match shards.set(k.as_bytes(), &args[1], DEFAULT_DURABILITY).await {
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
        match shards.get(k.as_bytes()).await {
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
        match shards.get(k.as_bytes()).await {
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
    incr_apply(k.as_bytes(), sign, shards).await
}

async fn incr_apply(key: &Bytes, delta: i64, shards: &ShardSet) -> Frame {
    let current: i64 = match shards.get(key).await {
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
    match shards.set(key, &body, DEFAULT_DURABILITY).await {
        Ok(()) => Frame::Integer(new),
        Err(e) => shard_error(&e),
    }
}

#[cfg(test)]
mod tests {
    /// Local deterministic tenant id from a string. Mirrors what the
    /// real tenant backend does (xxh3_128 of the name) for tests that
    /// need stable ids without pulling in a full backend impl.
    fn tid_from_name(name: &str) -> TenantId {
        let h = xxhash_rust::xxh3::xxh3_128(name.as_bytes());
        TenantId::from_bytes(h.to_le_bytes())
    }

    use super::*;
    use tempfile::TempDir;

    fn args(parts: &[&str]) -> Vec<Bytes> {
        parts
            .iter()
            .map(|s| Bytes::copy_from_slice(s.as_bytes()))
            .collect()
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

    #[tokio::test]
    async fn skeg_namespace_rejects_unknown_verb() {
        // Unknown `SKEG.*` verbs come through the parser as `Unknown`
        // and the dispatcher emits the legacy `ERR unknown command
        // 'SKEG.<verb>'` byte-for-byte.
        let (_dir, shards) = fresh_shards().await;
        let resp = dispatch_unknown(
            "SKEG.WHATEVER",
            vec![],
            &shards,
            TenantId::ZERO,
            None,
        )
        .await;
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
