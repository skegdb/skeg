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

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tracing::{debug, warn};

use skeg_core::{BoundedGet, Durability};
use skeg_resp3::{
    Command, ConnectionState, Frame, FrameDecoder, encode_frame, handle_echo, handle_ping,
    parse_command,
};
use skeg_vector::QuantKind;

use crate::failpoint::IngressFailpoint;
use crate::ingress::ConnectionBudget;
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
/// Longest accepted index name. Generous, but bounds the on-disk `vindex-<name>`
/// path and the registry key.
const MAX_VINDEX_NAME_LEN: usize = 255;

/// Delay applied before returning a failed-auth error, to throttle online
/// password guessing on the HELLO/AUTH path (which bypasses the QoS gate).
const AUTH_FAIL_PENALTY: Duration = Duration::from_secs(1);

/// Rolling window over which failed auth attempts from one source IP are
/// counted, and the count that trips the block for the rest of the window.
/// HELLO/AUTH bypass the QoS admission gate, so this shared per-IP counter is
/// the throttle that survives reconnects (the per-connection tarpit alone does
/// not): a flood that opens a fresh connection per guess still lands here.
const AUTH_FAIL_WINDOW: Duration = Duration::from_secs(30);
const AUTH_FAIL_MAX: u32 = 5;

struct AuthFail {
    count: u32,
    window_start: Instant,
}

/// Failed-auth counters keyed by source IP. Process-global so every connection
/// task shares it. Small: only IPs with recent failures, pruned as they expire.
fn auth_failures() -> &'static Mutex<HashMap<IpAddr, AuthFail>> {
    static M: OnceLock<Mutex<HashMap<IpAddr, AuthFail>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

/// True if `ip` has hit `AUTH_FAIL_MAX` failures inside the current window.
fn auth_is_blocked(ip: IpAddr) -> bool {
    let now = Instant::now();
    let map = auth_failures().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ip).is_some_and(|r| {
        now.duration_since(r.window_start) < AUTH_FAIL_WINDOW && r.count >= AUTH_FAIL_MAX
    })
}

/// Record one failed attempt from `ip`, starting a fresh window if the last one
/// elapsed. Opportunistically drops entries whose window has expired so the map
/// cannot grow without bound.
fn auth_record_failure(ip: IpAddr) {
    let now = Instant::now();
    let mut map = auth_failures().lock().unwrap_or_else(|e| e.into_inner());
    map.retain(|_, r| now.duration_since(r.window_start) < AUTH_FAIL_WINDOW);
    let entry = map.entry(ip).or_insert(AuthFail {
        count: 0,
        window_start: now,
    });
    if now.duration_since(entry.window_start) >= AUTH_FAIL_WINDOW {
        entry.count = 0;
        entry.window_start = now;
    }
    entry.count = entry.count.saturating_add(1);
}

/// Clear an IP's counter after a successful login.
fn auth_clear(ip: IpAddr) {
    auth_failures()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&ip);
}

/// Per-connection input-buffer ceiling. The parser only yields a frame once
/// the whole aggregate is buffered, so this is a *frame* cap, not a bulk cap:
/// the largest legitimate frame is a `SKEG.VMSET` at its vector cap
/// (`MAX_VMSET_BYTES`) plus its ids and payload bulks, each bounded by
/// `MAX_BULK_LEN`, plus framing headroom. Caps how much a single connection
/// can pin while a frame is mid-flight, so a desynced or dribbled
/// never-completing frame cannot grow the buffer without bound (and N
/// connections cannot each pin more than this).
pub(crate) const MAX_CONN_BUFFER: usize = MAX_VMSET_BYTES + skeg_resp3::MAX_BULK_LEN + (1 << 20);

fn scope_vindex_or_reject(tenant: TenantId, raw_name: &str) -> Result<String, Frame> {
    if raw_name.contains("::") {
        return Err(Frame::Error(
            "ERR VINDEX name must not contain '::' (reserved for tenant scoping)".into(),
        ));
    }
    // The name flows into `dir.join(format!("vindex-{name}"))` for create /
    // File::create / remove_dir_all. Without this, a crafted name escapes the
    // data dir: VINDEX.CREATE "../../x" writes outside it and VINDEX.DROP
    // "../../victim" would recursively delete an arbitrary writable dir. Allow
    // only a safe filename charset; reject empties, over-long, and any path
    // separator, parent ref, or control byte.
    let ok = !raw_name.is_empty()
        && raw_name.len() <= MAX_VINDEX_NAME_LEN
        && raw_name != "."
        && raw_name != ".."
        && !raw_name.contains("..")
        && raw_name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'));
    if !ok {
        return Err(Frame::Error(
            "ERR VINDEX name: 1-255 chars, [A-Za-z0-9._-] only, no '..'".into(),
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

/// How much spare decoder capacity to reserve before the next socket read.
///
/// A few KiB while idle or between frames (`buffered == 0`): an accepted-but-
/// silent connection must not pin real memory just for existing, and the
/// connection semaphore's default limit means there can be hundreds of these
/// at once. Once a frame is mid-flight (`buffered > 0` - a partial read left
/// bytes the decoder hasn't parsed yet), reserve the large chunk so a
/// pipelined burst still buffers many frames per syscall instead of
/// serializing one-frame-per-read.
pub(crate) fn read_reserve(buffered: usize) -> usize {
    if buffered == 0 { 4096 } else { 256 * 1024 }
}

/// Give back the large chunk once the buffer is fully drained. `BytesMut`
/// keeps its allocation across `split_to`, so without this a connection that
/// bursted once - or sent a single byte and went quiet - would hold 256 KiB
/// for its whole life, and `read_reserve`'s idle figure would only be true
/// for sockets that never sent anything.
pub(crate) fn trim_idle(buf: &mut BytesMut) {
    if buf.is_empty() && buf.capacity() > 64 * 1024 {
        *buf = BytesMut::with_capacity(4096);
    }
}

const SKEG_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default durability for SET / DEL. `Kernel` survives process+kernel crash
/// without `F_FULLFSYNC` cost - same default as the binary protocol handler.
const DEFAULT_DURABILITY: Durability = Durability::Kernel;

/// Server-assigned connection id, monotonic across the process lifetime.
/// Exposed via HELLO response and (future) CLIENT ID.
static CONN_COUNTER: AtomicI64 = AtomicI64::new(1);

/// How much slack `frame_upper_bound` adds around one frame node's own
/// bytes: the type byte, the RESP length prefix, and the trailing CRLF (both
/// ends, for an aggregate). No legitimate skeg reply approaches this many
/// digits of length - it is chosen generous rather than exact, because it
/// bounds a RESERVATION, not the wire; `encode_frame` remains the only
/// source of truth for what is actually written.
const FRAME_NODE_OVERHEAD: usize = 32;

/// A tight upper bound on what `encode_frame` will write for `frame`,
/// computed by walking the `Frame` tree that dispatch already built -
/// without encoding it a second time. `Frame` already owns every byte it
/// will emit (a `Bulk`'s `Bytes`, an `Array`'s child frames), so THIS sum is
/// over data that is already resident, not a second fetch.
///
/// That is the limit of what this function buys, and it matters to say so
/// precisely: by the time a `Frame` exists, the store has already answered -
/// "no longer speculative" describes the SIZE, measured instead of
/// estimated, and says nothing about whether the bytes are already
/// allocated, because they are. `frame_upper_bound` reserves before
/// `encode_frame`'s buffer, the SECOND allocation a reply makes; it does
/// nothing about the FIRST, the fetch that built `Frame` in the first place
/// (`shards.vsearch` materialising every hit's payload as `Bytes`, before
/// this function or `flush_reply` ever runs). Reserving the first requires a
/// bound computable BEFORE that fetch - `reply_upper_bound` covers this for
/// every command whose worst case a request can name (`k`, `WITHPAYLOAD`, a
/// key count) - a mutation's bound additionally has to precede its commit,
/// a read's does not, but both need it before the store call, not after.
fn frame_upper_bound(frame: &Frame) -> usize {
    let payload = match frame {
        Frame::Simple(s) => s.len(),
        Frame::Error(s) => s.len(),
        Frame::Integer(_) => 20,
        Frame::Bulk(b) => b.len(),
        Frame::Null => 0,
        Frame::Array(items) | Frame::Set(items) | Frame::Push(items) => {
            items.iter().map(frame_upper_bound).sum()
        }
        Frame::BlobError { code, message } => code.len() + message.len() + 1,
        Frame::Boolean(_) => 1,
        Frame::Double(_) => 32,
        Frame::Map(pairs) => pairs
            .iter()
            .map(|(k, v)| frame_upper_bound(k) + frame_upper_bound(v))
            .sum(),
        Frame::Verbatim { data, .. } => data.len() + 4,
        Frame::BigNumber(s) => s.len(),
    };
    payload.saturating_add(FRAME_NODE_OVERHEAD)
}

/// Encode one reply into `out`, write it, and account for what it cost.
///
/// The reply buffer is a per-connection buffer exactly like the decoder's,
/// and it used to be in no budget at all - see `frame_upper_bound`, which
/// this reserves BEFORE `encode_frame` runs rather than after, so the
/// governor sees the peak coming instead of reading it off a `BytesMut` that
/// already grew to hold it. `other_capacity` is what the rest of this
/// connection holds (the decoder's buffer, plus any reply still queued
/// ahead of this one in the pipeline): the charge is the connection's whole
/// footprint, ingress and egress in one figure, because they are one
/// socket's memory.
///
/// A reservation taken here can still be refused - the bound is an
/// ESTIMATE, taken without knowing whether other connections have since
/// filled the class - and when that happens the reply is written anyway and
/// the overshoot is COUNTED, not returned: by the time a `Frame` exists the
/// work behind it has already committed (a mutation's caller reserved its
/// own bound in `handle_connection_resp3` before that commit ran; a read
/// commits nothing), and withdrawing the answer now would be an error
/// raised past the point where undoing it is possible.
///
/// `frame_upper_bound` bounds the CONTENT `encode_frame` writes, not the
/// CAPACITY `out` ends up with - `BytesMut`'s growth doubles past what it
/// needs, so `out.capacity()` routinely overshoots the content it holds.
/// This does not reserve for that slack: the budget already charges
/// capacity, not length, at every OTHER call site for exactly the reason the
/// module doc gives (the allocation is what the process pays for), and here
/// too the true peak is `out.capacity()` - but re-reserving for it after the
/// fact would mean charging the SAME growth twice, once as an estimate and
/// once as the real thing, which only shrinks how much of the connection's
/// allowance is left for what happens next. `trim_idle` frees the real
/// allocation immediately after the write regardless of what the budget
/// charged for it, so the memory itself is never the thing left unaccounted.
async fn flush_reply(
    stream: &mut (impl tokio::io::AsyncWrite + Unpin),
    out: &mut BytesMut,
    frame: &Frame,
    version: skeg_resp3::ProtoVersion,
    budget: &mut ConnectionBudget,
    other_capacity: usize,
) -> bool {
    if budget
        .grow_to(other_capacity.saturating_add(frame_upper_bound(frame)))
        .is_err()
    {
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::IngressReplyOverBudget);
    }
    out.clear();
    encode_frame(frame, version, out);
    let ok = stream.write_all(out).await.is_ok();
    // Given back immediately, not at the next read: `trim_idle` only fires on
    // an EMPTY buffer, and `out` is only empty between replies - which is
    // exactly now.
    out.clear();
    trim_idle(out);
    budget.shrink_to(other_capacity.saturating_add(out.capacity()));
    ok
}

/// Per-connection driver. Loops until EOF / write error / fatal parse error.
///
/// `budget` is this connection's share of the ingress class, taken at accept
/// and held for the whole life of the connection. `fp_key` is the listener's
/// port, which is what an ingress failpoint is keyed on.
pub async fn handle_connection_resp3(
    mut stream: TcpStream,
    shards: ShardSet,
    tenant_backend: Option<Arc<dyn TenantBackend>>,
    mut budget: ConnectionBudget,
    fp_key: Arc<str>,
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
    // Each in-flight command carries the bytes reserved against `budget` for
    // ITS reply (0 for one with no computable pre-dispatch bound - a read,
    // whose reservation instead happens in `flush_reply`). `reply_reserved`
    // is the running sum of those, because `ConnectionBudget::grow_to` is a
    // "hold at least this much" call, not an additive one: without tracking
    // the sum here, reserving before each of several pipelined `SKEG.VMSET`s
    // would only ever charge for the LARGEST of them, and the other 127
    // slots in the window would be free to fan out unreserved fan-out
    // replies - the exact multiplication P0-A is about, just moved one level
    // down, from connections to one connection's pipeline.
    let mut inflight: VecDeque<(tokio::task::JoinHandle<Frame>, u64)> = VecDeque::new();
    let mut reply_reserved: u64 = 0;
    // Await the oldest in-flight command, release its share of
    // `reply_reserved`, and write its response in order. Returns false on
    // write failure (caller must stop).
    macro_rules! emit_front {
        () => {{
            let mut ok = true;
            if let Some((h, reserved)) = inflight.pop_front() {
                reply_reserved = reply_reserved.saturating_sub(reserved);
                let resp = h
                    .await
                    .unwrap_or_else(|_| Frame::Error("ERR internal task failure".into()));
                let held = decoder.capacity().saturating_add(reply_reserved as usize);
                ok = flush_reply(
                    &mut stream,
                    &mut out,
                    &resp,
                    state.version,
                    &mut budget,
                    held,
                )
                .await;
            }
            ok
        }};
    }

    'conn: loop {
        match decoder.decode() {
            Ok(Some(frame)) => match parse_command(frame) {
                Ok(cmd) if is_pipelineable(&cmd) => {
                    // RESERVE BEFORE COMMIT: a bound computable from the
                    // request is reserved (summed with everything already
                    // queued ahead of it) before the command - which may be
                    // a mutation - is allowed to run at all. A refusal here
                    // never undoes work, because no work has happened yet.
                    let this_bound = reply_upper_bound(&cmd).unwrap_or(0) as u64;
                    let refused = if this_bound > 0 {
                        let want = decoder
                            .capacity()
                            .saturating_add((reply_reserved + this_bound) as usize);
                        match budget.grow_to(want) {
                            Ok(()) => {
                                reply_reserved += this_bound;
                                None
                            }
                            Err(e) => Some(e),
                        }
                    } else {
                        None
                    };
                    let handle = match refused {
                        None => {
                            let (sh, be, t) = (shards.clone(), tenant_backend.clone(), tenant);
                            (tokio::spawn(exec_pipelined(cmd, t, sh, be)), this_bound)
                        }
                        Some(e) => {
                            skeg_telemetry::tick_counter(
                                skeg_telemetry::Counter::IngressRefusedGrowth,
                            );
                            let msg = crate::admission::AdmissionError::from(e).wire_message();
                            (tokio::spawn(async move { Frame::Error(msg) }), 0)
                        }
                    };
                    inflight.push_back(handle);
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
                            // Same reserve-before-commit as the pipelined
                            // arm above; `reply_reserved` is 0 here (the
                            // drain just above emptied it), so the bound
                            // reserved is this command's alone.
                            match reply_upper_bound(&cmd) {
                                Some(bound) if bound > 0 => {
                                    let want = decoder.capacity().saturating_add(bound);
                                    if let Err(e) = budget.grow_to(want) {
                                        skeg_telemetry::tick_counter(
                                            skeg_telemetry::Counter::IngressRefusedGrowth,
                                        );
                                        Frame::Error(
                                            crate::admission::AdmissionError::from(e)
                                                .wire_message(),
                                        )
                                    } else {
                                        // The bound above already covers this
                                        // command's reply; a KV read is not
                                        // one of them (`reply_upper_bound`
                                        // answers `None` for `GET`/`MGET`).
                                        dispatch_command(
                                            cmd,
                                            &mut state,
                                            &mut tenant,
                                            &shards,
                                            tenant_backend.as_ref(),
                                            peer.map(|p| p.ip()),
                                            None,
                                        )
                                        .await
                                    }
                                }
                                _ => {
                                    // Where `GET`/`MGET` land. They cannot be
                                    // sized from the request, so they carry
                                    // the budget INTO the dispatch and reserve
                                    // there, from the lengths the index holds,
                                    // before the first value is read.
                                    let held = decoder.capacity();
                                    dispatch_command(
                                        cmd,
                                        &mut state,
                                        &mut tenant,
                                        &shards,
                                        tenant_backend.as_ref(),
                                        peer.map(|p| p.ip()),
                                        Some(&mut ReadAdmission::new(&mut budget, held)),
                                    )
                                    .await
                                }
                            }
                        }
                        Err(e) => Frame::Error(format!("ERR {e}")),
                    };
                    let held = decoder.capacity();
                    if !flush_reply(
                        &mut stream,
                        &mut out,
                        &response,
                        state.version,
                        &mut budget,
                        held,
                    )
                    .await
                    {
                        break 'conn;
                    }
                }
            },
            Ok(None) => {
                // Buffer drained: emit the in-flight burst (bounds latency + lets
                // its payload writes group-commit together), then read more.
                //
                // Held only under a failpoint, and only so a test can decide
                // WHEN the pipeline drains instead of inheriting whatever the
                // kernel did with the peer's writes. Everything below already
                // accounts for `reply_reserved`, so a held pipeline keeps its
                // reservations rather than handing them back under itself.
                if !crate::fp_ingress!(IngressFailpoint::HoldPipelineDrain, &fp_key) {
                    while !inflight.is_empty() {
                        if !emit_front!() {
                            break 'conn;
                        }
                    }
                }
                // Pull a large chunk per syscall so a pipelined burst buffers
                // many frames at once (the default 4 KiB spare = one ~4 KiB VSET
                // frame, which would serialize the pipeline one-frame-per-read).
                // But only once a frame is actually mid-flight: reserving the
                // large chunk unconditionally means an idle connection - most
                // of them, under the connection semaphore's default limit -
                // pins that capacity for nothing. An idle or between-frames
                // socket gets a few KiB instead; a pipelined burst still grows
                // to the large chunk after its first partial read fills the
                // small reservation and leaves bytes buffered.
                trim_idle(decoder.buf_mut());
                // Give back what the trim released BEFORE asking for more:
                // a connection that bursted once must not still be charged for
                // its peak while it asks for the next chunk. `reply_reserved`
                // is added because `shrink_to` is a "hold at most this" call
                // and the replies still in flight are holding their own share
                // of the charge - normally zero here, since the drain above
                // just emptied the pipeline, but never negative and never a
                // reservation given back under a reply that still needs it.
                let queued = reply_reserved as usize;
                budget.shrink_to(decoder.buf_mut().capacity().saturating_add(queued));
                let reserve = read_reserve(decoder.buffered());
                // What `BytesMut::reserve` will leave the capacity at. It never
                // shrinks and it grows to at least len + additional, so this is
                // the charge to hold BEFORE the buffer is allowed to get there.
                let want = decoder
                    .buf_mut()
                    .capacity()
                    .max(decoder.buffered().saturating_add(reserve))
                    .saturating_add(queued);
                if let Err(e) = crate::ingress::grow_or_stall(&mut budget, want, &fp_key).await {
                    // Back to the floor before the refusal goes out: the bytes
                    // of this frame that did arrive are not worth keeping for a
                    // frame that will not be completed, and holding them would
                    // make the refusal cost what admitting it would have.
                    *decoder.buf_mut() = BytesMut::with_capacity(4096);
                    budget.shrink_to(4096);
                    skeg_telemetry::tick_counter(skeg_telemetry::Counter::IngressRefusedGrowth);
                    warn!(?peer, "ingress refused: {e}");
                    let err = Frame::Error(e.wire_message());
                    let _ = flush_reply(
                        &mut stream,
                        &mut out,
                        &err,
                        state.version,
                        &mut budget,
                        4096,
                    )
                    .await;
                    break 'conn;
                }
                decoder.buf_mut().reserve(reserve);
                match stream.read_buf(decoder.buf_mut()).await {
                    Ok(0) => break,
                    Ok(_) => {
                        // Re-checked AFTER the read. `read_buf` fills the whole
                        // spare capacity, and `BytesMut::chunk_mut` will add a
                        // little more when the buffer is exactly full, so the
                        // capacity charged for a moment ago is not necessarily
                        // the capacity that came back.
                        if let Err(e) =
                            budget.grow_to(decoder.buf_mut().capacity().saturating_add(queued))
                        {
                            *decoder.buf_mut() = BytesMut::with_capacity(4096);
                            budget.shrink_to(4096);
                            skeg_telemetry::tick_counter(
                                skeg_telemetry::Counter::IngressRefusedGrowth,
                            );
                            warn!(?peer, "ingress refused after read: {e}");
                            let err = Frame::Error(e.wire_message());
                            let _ = flush_reply(
                                &mut stream,
                                &mut out,
                                &err,
                                state.version,
                                &mut budget,
                                4096,
                            )
                            .await;
                            break 'conn;
                        }
                        // Bound per-connection buffering. A frame that never
                        // completes (protocol desync, or a bulk whose declared
                        // length is dribbled forever) would otherwise let one
                        // connection pin ~129 MiB, and N connections N× that.
                        // One max-size bulk plus headroom is the legitimate
                        // ceiling; past it the peer is misbehaving.
                        if decoder.buffered() > MAX_CONN_BUFFER {
                            warn!(?peer, "input buffer ceiling exceeded, closing");
                            break;
                        }
                        continue;
                    }
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
                let held = decoder.capacity();
                let _ = flush_reply(
                    &mut stream,
                    &mut out,
                    &err,
                    state.version,
                    &mut budget,
                    held,
                )
                .await;
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

    // The budget goes back HERE, and the failpoint marks that this line was
    // reached. Without it a test asserting "the class came back to zero" would
    // pass just as well on a connection that was never served at all.
    if crate::fp_ingress!(IngressFailpoint::ReleaseDeferredOnClose, &fp_key) {
        tokio::task::yield_now().await;
    }
    drop(budget);
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
        // Each walks every key on every shard (and reclaim rewrites segments);
        // charge a flat premium so the admission gate throttles them, not the
        // per-key work which is not known from the args.
        Command::SkegSubjectErase { .. }
        | Command::SkegTenantErase { .. }
        | Command::SkegTenantDelete { .. } => 100,
        Command::SkegReclaim => 1000,
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
        | Command::Append { .. }
        | Command::Mset { .. }
        | Command::Del { .. }
        | Command::Incr { .. }
        | Command::Decr { .. }
        | Command::IncrBy { .. }
        | Command::DecrBy { .. } => CommandKind::KvWrite,
        Command::SkegVsearch { .. } | Command::SkegVget { .. } | Command::SkegVgraph { .. } => {
            CommandKind::VectorRead
        }
        Command::SkegVset { .. } | Command::SkegVmset { .. } | Command::SkegVdel { .. } => {
            CommandKind::VectorWrite
        }
        Command::SkegVindexCreate { .. } => CommandKind::VindexCreate,
        Command::SkegVindexReshard { .. } | Command::SkegVindexOverlap { .. } => {
            CommandKind::VectorWrite
        }
        Command::SkegVindexDrop { .. } => CommandKind::VindexDrop,
        Command::SkegVindexConsolidate { .. } => CommandKind::VindexConsolidate,
        Command::SkegVindexList => CommandKind::VindexList,
        Command::SkegCheck { .. } => CommandKind::VindexList,
        Command::SkegVowner { .. } => CommandKind::VindexList,
        Command::SkegHealth { .. } => CommandKind::VindexList,
        Command::SkegVindexShards { .. } => CommandKind::VindexList,
        // A tenant erasing its own subject's keys is a KV write; the two
        // cross-tenant / store-wide ops are admin.
        Command::SkegSubjectErase { .. } => CommandKind::KvWrite,
        Command::SkegTenantErase { .. }
        | Command::SkegTenantDelete { .. }
        | Command::SkegReclaim
        | Command::SkegQuotaSet { .. }
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
            | Command::SkegVget { .. }
            | Command::SkegVgraph { .. }
            | Command::Ping(_)
            | Command::Echo(_)
    )
}

/// A reply this small covers `+OK`, an integer, a short array of integers,
/// or a bounded error line - which is every mutation on this wire except
/// `SKEG.VMSET` (see [`vmset_reply_upper_bound`]). Backed by
/// `every_bounded_mutations_reply_fits_the_small_bound`, which dispatches a
/// sample of each at worst-case argument sizes (the longest legal vindex
/// name, a full tenant id) and measures the encoded reply.
const SMALL_REPLY_BOUND: usize = 4096;

/// The upper bound of a `SKEG.VMSET` reply, computed from the request alone
/// (before a single item has run) by the same arithmetic `skeg_vmset` uses
/// to count items, capped at [`MAX_VMSET_ITEMS`] the same way the admission
/// check there is. One reply line per item, each at most
/// [`MAX_VMSET_ERROR_LEN`] (`skeg_vmset` caps every item's error to that via
/// `cap_item_error`), plus per-node framing.
///
/// An arity this function does not recognise (not `1 + 3n` args) still gets
/// a finite answer rather than a panic: `skeg_vmset` itself refuses that
/// request with a short error frame, which fits comfortably under any bound
/// this returns.
fn vmset_reply_upper_bound(args: &[Bytes]) -> usize {
    let n_items = (args.len().saturating_sub(1) / 3).min(MAX_VMSET_ITEMS);
    n_items
        .saturating_mul(MAX_VMSET_ERROR_LEN.saturating_add(FRAME_NODE_OVERHEAD))
        .saturating_add(FRAME_NODE_OVERHEAD)
}

/// The largest vector `SKEG.VGET` may ever answer with, in bytes of
/// little-endian `f32`.
///
/// The honest bound is the INDEX'S OWN dim, but nothing in `ShardSet` reads
/// it synchronously today without an async fan-out (`vindex_list`) or a
/// router that only exists once an index has been trained
/// (`ShardSet::router`) - neither is a per-request-shape bound, and adding a
/// synchronous dim registry touches vindex create/drop/reopen in `shard.rs`,
/// which is out of THIS mandate's reach (see the R1 handoff: other branches
/// are mid-flight on that file). So this is the documented-constant fallback
/// the mandate allows when a tighter bound needs a change bigger than the
/// bound itself: 1 MiB is `262_144` `f32`s, an order of magnitude past any
/// embedding dimension in production use, and the same shape of ceiling
/// `MAX_VGET_VECTOR_BYTES` names so nobody has to rediscover the number.
pub(crate) const MAX_VGET_VECTOR_BYTES: usize = 1024 * 1024;

pub(crate) fn vget_reply_upper_bound() -> usize {
    MAX_VGET_VECTOR_BYTES.saturating_add(FRAME_NODE_OVERHEAD)
}

/// The upper bound of a `SKEG.VSEARCH` reply, computed from the request
/// alone: `k` hits (clamped to [`crate::shard::MAX_VSEARCH_K`] the same way
/// the shard clamps it), each an id (a `u64` printed as decimal, at most 20
/// digits), a score, and - only when `WITHPAYLOAD` is present - one payload
/// blob at its staged ceiling ([`MAX_PAYLOAD_BYTES`], enforced at
/// `SKEG.VSET`/`SKEG.VMSET` staging so this bound is never a guess about
/// what a write was allowed to store). `skeg_vsearch` builds exactly this
/// shape per hit: `(id, score[, payload])`.
///
/// An arity or `k` this cannot parse still gets a finite, conservative
/// answer (`k` defaults to the maximum) rather than a panic: `skeg_vsearch`
/// itself refuses a malformed request with a short error frame, which fits
/// under any bound this returns.
fn vsearch_reply_upper_bound(args: &[Bytes]) -> usize {
    let k = args
        .get(1)
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(crate::shard::MAX_VSEARCH_K)
        .min(crate::shard::MAX_VSEARCH_K);
    let with_payload = args
        .get(4..)
        .is_some_and(|tail| tail.iter().any(|a| a.eq_ignore_ascii_case(b"WITHPAYLOAD")));
    let per_hit = 20usize // id, decimal u64
        .saturating_add(FRAME_NODE_OVERHEAD)
        .saturating_add(FRAME_NODE_OVERHEAD) // score
        .saturating_add(if with_payload {
            MAX_PAYLOAD_BYTES.saturating_add(FRAME_NODE_OVERHEAD)
        } else {
            0
        });
    k.saturating_mul(per_hit)
        .saturating_add(FRAME_NODE_OVERHEAD)
}

/// The largest out-degree a Vamana graph node in this engine can carry.
///
/// Mirrors `skeg_vector::vamana::MAX_R` (currently 64), which is private to
/// that crate and not re-exported - reaching it would mean widening
/// `skeg-vector`'s public API for one constant, a change to a crate other
/// branches are mid-flight on (see the R1/A1 handoffs) and bigger than the
/// bound it would tighten. Documented here instead, the same
/// documented-constant fallback `MAX_VGET_VECTOR_BYTES` already uses: if
/// `MAX_R` ever changes, this drifts silently rather than failing to
/// compile, which is the trade a private upstream constant forces.
const VGRAPH_MAX_OUT_DEGREE: usize = 64;

/// One `SKEG.VGRAPH` node line (`n <id> <degree>\n`) or edge line
/// (`e <a> <b>\n`), generous: a `u64` printed in decimal is at most 20
/// digits, and neither line carries more than two of them.
const VGRAPH_LINE_BYTES: usize = 48;

/// The upper bound of a `SKEG.VGRAPH` reply, computed from the request
/// alone: `skeg_vgraph` clamps `count` to `[1, 2048]` before it ever reaches
/// `shards.graph_sample`, and samples at most `count` nodes, each
/// contributing at most one node line and [`VGRAPH_MAX_OUT_DEGREE`] edge
/// lines (a Vamana node's out-degree is capped at build time, never
/// exceeded at read time). `SkegVgraph` is pipelineable
/// (`is_pipelineable`), so - same as `SkegVsearch` - this is what keeps a
/// burst of them from sitting in `inflight` fully built and uncharged: a
/// typical degree (~64) at `count=2048` is on the order of several MiB per
/// reply, and `PIPELINE_WINDOW` (128) of those uncharged is the same
/// hundreds-of-MiB shape P0-A names for `SKEG.VMSET`.
///
/// An unparseable `count` (or none at all - `skeg_vgraph` defaults it to
/// 120) still gets a finite, conservative answer: the clamp's own
/// ceiling, 2048.
fn vgraph_reply_upper_bound(args: &[Bytes]) -> usize {
    let count = args
        .get(1)
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
        .clamp(1, 2048);
    count
        .saturating_mul(VGRAPH_LINE_BYTES)
        .saturating_add(
            count
                .saturating_mul(VGRAPH_MAX_OUT_DEGREE)
                .saturating_mul(VGRAPH_LINE_BYTES),
        )
        .saturating_add(FRAME_NODE_OVERHEAD)
}

/// The upper bound of `cmd`'s reply, computable from the REQUEST alone -
/// before the command has run, before `shards.*` is ever called. Reserving
/// this BEFORE dispatch is what P0-A actually asks for: the allocation must
/// not precede the budget, for a read every bit as much as a mutation. A
/// mutation's own reservation additionally has to precede its COMMIT
/// (refusing after the write already happened would tell a client to retry
/// work that is done) - that is a stronger requirement a mutation carries on
/// top of this one, not a different reason for reads to skip it.
///
/// `None` means the worst case cannot be sized from the request alone - it
/// depends on how much the STORE holds (an index count, a shard count, a
/// graph sample's edges) rather than on a number the client supplied (a `k`,
/// a `WITHPAYLOAD` flag, a key count). For those, and ONLY those, the
/// allocation genuinely cannot be reserved for before it happens on this
/// server, and the bound instead falls to `frame_upper_bound` measuring the
/// `Frame` dispatch already built, right before `encode_frame` runs in
/// [`flush_reply`] - which bounds the SECOND allocation (the encoder's
/// buffer), not the first (the fetch), and is not a substitute for a
/// pre-dispatch reservation where one is computable.
///
/// Exhaustive over [`Command`], no `_` arm: a new variant has to say which
/// side of that line it is on rather than defaulting onto either one.
fn reply_upper_bound(cmd: &Command) -> Option<usize> {
    match cmd {
        // ---- mutations: reserved before dispatch runs, because dispatch is
        // where these commit. Everything here answers `+OK`, an integer, a
        // short array of integers, or a bounded error line, regardless of
        // the request's own size - except `SKEG.VMSET`, whose bound is a
        // request-derived formula, not a constant. ----
        Command::Set { .. }
        | Command::Append { .. }
        | Command::Del { .. }
        | Command::Mset { .. }
        | Command::Incr { .. }
        | Command::Decr { .. }
        | Command::IncrBy { .. }
        | Command::DecrBy { .. }
        | Command::SkegVindexCreate { .. }
        | Command::SkegVindexDrop { .. }
        | Command::SkegVindexConsolidate { .. }
        | Command::SkegVset { .. }
        | Command::SkegVdel { .. }
        | Command::SkegVindexReshard { .. }
        | Command::SkegVindexOverlap { .. }
        | Command::SkegSubjectErase { .. }
        | Command::SkegTenantErase { .. }
        | Command::SkegTenantDelete { .. }
        | Command::SkegReclaim
        | Command::SkegQuotaSet { .. }
        | Command::SkegQosSet { .. } => Some(SMALL_REPLY_BOUND),
        Command::SkegVmset { args } => Some(vmset_reply_upper_bound(args)),

        // ---- reads whose worst-case reply IS computable from the request
        // AND that are pipelineable - which together is what makes a
        // pre-dispatch reservation both possible and necessary. P0-A is
        // about the allocation preceding the budget, not about a commit
        // needing protection: a `SKEG.VSEARCH k=4096 WITHPAYLOAD` materialises
        // its hits and their payloads (id, score, up to a megabyte of blob
        // each) into a `Frame` BEFORE `frame_upper_bound` ever runs on it, so
        // measuring the frame after the fact catches the SECOND allocation
        // (`encode_frame`'s buffer) and misses the first (the fetch itself).
        // Because `SkegVget`/`SkegVsearch`/`SkegVgraph` are pipelineable
        // (`is_pipelineable`), a `Some` bound here also sums into
        // `reply_reserved` for the pipeline window - closing audit 19 A1's
        // reproduced hazard: up to `PIPELINE_WINDOW` completed,
        // payload-bearing reads sitting in `inflight` uncharged (measured at
        // 20,809,984 bytes under a 4 MiB class, 100 pipelined
        // `SKEG.VSEARCH k=50 WITHPAYLOAD`), and its A2 twin for `SKEG.VGRAPH`
        // (count clamped to 2048, a typical Vamana out-degree of ~64 makes
        // one reply on the order of several MiB - the same hundreds-of-MiB
        // shape under `PIPELINE_WINDOW`, left open by round 1 with no
        // structural reason `VGET`/`VSEARCH` did not share). ----
        Command::SkegVget { .. } => Some(vget_reply_upper_bound()),
        Command::SkegVsearch { args } => Some(vsearch_reply_upper_bound(args)),
        Command::SkegVgraph { args } => Some(vgraph_reply_upper_bound(args)),

        // ---- everything else commits nothing on the way to answering, is
        // NOT pipelineable (`Get`/`Mget`/`Exists` are deliberately excluded -
        // see `is_pipelineable`), or has no bound computable from the
        // request alone - in every one of those cases a pre-dispatch
        // reservation buys nothing `frame_upper_bound` does not already give
        // it, and can cost real correctness. ----
        //
        // `Get`/`Mget`: no pipeline accumulation is possible for them (never
        // pipelined, so never more than one such reply outstanding on a
        // connection at a time - not the hazard above), and the one bound
        // computable from the request alone is the wire's own ceiling on a
        // stored value (`MAX_BULK_LEN`, 64 MiB) - which, multiplied by even
        // a handful of `MGET` keys, refuses ordinary requests under any
        // realistically sized class (confirmed: it broke RESP3 conformance's
        // own `kv.mget.order.with.hole`, a four-key MGET, needing >512 MiB
        // against a ~270 MiB connection allowance). A bound this far from
        // the common case is not a safety margin, it is a different feature.
        // `frame_upper_bound` already reserves their real, fetched size
        // before `encode_frame` runs - the literal P0-A fix - and no
        // accumulation multiplies a single barrier-path GET's cost.
        //
        // `Exists`'s reply is a bare integer regardless of key count.
        // `Hello`/`Select`/`SkegAuth`/`SkegWhoami`/the quota-and-QoS
        // getters/`Unknown`/`Ping`/`Echo` are small but do not need a
        // SEPARATE reservation on top of what they already hold - adding one
        // pushed an idle connection sitting exactly at its floor a few bytes
        // over it, which a genuinely saturated class then refused, and
        // starving an existing connection on a class that is full is the one
        // thing admission must not do.
        //
        // `SkegStats`/`SkegShards`/`SkegVindexList`/`SkegCheck`/`SkegVowner`/
        // `SkegHealth`/`SkegVindexShards` are sized by how much the STORE
        // holds (index count, shard count), not by anything in the request -
        // found-not-fixed by this pass, lower severity than `VGET`/
        // `VSEARCH`/`VGRAPH` because none of them scale with an
        // attacker-chosen per-request multiplier the way `k`/`count` or a
        // payload blob does, AND none of them is pipelineable
        // (`is_pipelineable`) - so, like `Get`/`Mget`, no accumulation
        // multiplies a single reply's cost. Their bound is instead measured,
        // exactly, from the `Frame` dispatch built - see `frame_upper_bound`
        // in `flush_reply`. ----
        Command::Get { .. }
        | Command::Mget { .. }
        | Command::Hello(_)
        | Command::Select { .. }
        | Command::SkegWhoami
        | Command::SkegAuth { .. }
        | Command::SkegQuotaGet { .. }
        | Command::SkegQosGet { .. }
        | Command::Unknown { .. }
        | Command::Ping(_)
        | Command::Echo(_)
        | Command::Exists { .. }
        | Command::SkegStats
        | Command::SkegShards
        | Command::SkegVindexList
        | Command::SkegCheck { .. }
        | Command::SkegVowner { .. }
        | Command::SkegHealth { .. }
        | Command::SkegVindexShards { .. } => None,
    }
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
            Err(rejected) => {
                return Frame::Error(
                    crate::admission::AdmissionError::from_backend(rejected).wire_message(),
                );
            }
        },
    };
    let be = backend.as_ref();
    match cmd {
        Command::SkegVset { args } => skeg_vset(&args, &shards, tenant, be).await,
        Command::SkegVmset { args } => skeg_vmset(&args, &shards, tenant, be).await,
        Command::SkegVsearch { args } => skeg_vsearch(&args, &shards, tenant).await,
        Command::SkegVdel { args } => skeg_vdel(&args, &shards, tenant).await,
        Command::SkegVget { args } => skeg_vget(&args, &shards, tenant).await,
        Command::SkegVgraph { args } => skeg_vgraph(&args, &shards, tenant).await,
        Command::SkegVindexReshard { args } => skeg_vindex_reshard(&args, &shards, tenant).await,
        Command::SkegVindexOverlap { args } => skeg_vindex_overlap(&args, &shards, tenant).await,
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
    peer_ip: Option<IpAddr>,
    read_admission: Option<&mut ReadAdmission<'_>>,
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
                // Through the classification, not around it. The line the
                // client sees is unchanged - a backend writes its own,
                // complete - but it is now a refusal the engine has an
                // opinion about, which is what gives the native wire a byte
                // to send and the operator a counter when the backend's
                // message carries no code word at all.
                Err(rejected) => {
                    return Frame::Error(
                        crate::admission::AdmissionError::from_backend(rejected).wire_message(),
                    );
                }
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
                    Some((user, pass)) => {
                        // Block a source IP that has burst past AUTH_FAIL_MAX in
                        // the window before paying the Argon2 verify cost, so a
                        // reconnect flood cannot burn CPU. Still tarpit the reply.
                        if peer_ip.is_some_and(auth_is_blocked) {
                            tokio::time::sleep(AUTH_FAIL_PENALTY).await;
                            return Frame::Error(
                                "WRONGPASS too many failed attempts, try again later".into(),
                            );
                        }
                        match ctx.verify_login(user, pass.as_bytes()) {
                            Some(tid) => {
                                if let Some(ip) = peer_ip {
                                    auth_clear(ip);
                                }
                                *tenant = tid;
                            }
                            None => {
                                // Count the failure against the source IP and
                                // tarpit the reply. HELLO/AUTH bypass the QoS
                                // gate, so this is the only online-guessing
                                // throttle.
                                if let Some(ip) = peer_ip {
                                    auth_record_failure(ip);
                                }
                                tokio::time::sleep(AUTH_FAIL_PENALTY).await;
                                return Frame::Error(
                                    "WRONGPASS invalid username-password pair".into(),
                                );
                            }
                        }
                    }
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
            kv_get(
                std::slice::from_ref(&key),
                shards,
                *tenant,
                tenant_backend,
                read_admission,
            )
            .await
        }
        Command::Set { key, value } => kv_set(&[key, value], shards, *tenant, tenant_backend).await,
        Command::Append { key, value } => {
            kv_append(&[key, value], shards, *tenant, tenant_backend).await
        }
        Command::Del { keys } => kv_del(&keys, shards, *tenant, tenant_backend).await,
        Command::Exists { keys } => kv_exists(&keys, shards, *tenant, tenant_backend).await,
        Command::Mget { keys } => {
            kv_mget(&keys, shards, *tenant, tenant_backend, read_admission).await
        }
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
        Command::SkegCheck { args } => skeg_check(args.as_slice(), shards, *tenant).await,
        Command::SkegVowner { args } => skeg_vowner(args.as_slice(), shards, *tenant).await,
        Command::SkegHealth { args } => skeg_health(args.as_slice(), shards, *tenant).await,
        Command::SkegVindexShards { args } => {
            skeg_vindex_shards(args.as_slice(), shards, *tenant).await
        }
        Command::SkegVindexCreate { args } => skeg_vindex_create(&args, shards, *tenant).await,
        Command::SkegVindexDrop { args } => skeg_vindex_drop(&args, shards, *tenant).await,
        Command::SkegVindexConsolidate { args } => {
            skeg_vindex_consolidate(&args, shards, *tenant).await
        }
        Command::SkegVset { args } => skeg_vset(&args, shards, *tenant, tenant_backend).await,
        Command::SkegVmset { args } => skeg_vmset(&args, shards, *tenant, tenant_backend).await,
        Command::SkegVdel { args } => skeg_vdel(&args, shards, *tenant).await,
        Command::SkegVget { args } => skeg_vget(&args, shards, *tenant).await,
        Command::SkegVgraph { args } => skeg_vgraph(&args, shards, *tenant).await,
        Command::SkegVindexReshard { args } => skeg_vindex_reshard(&args, shards, *tenant).await,
        Command::SkegVindexOverlap { args } => skeg_vindex_overlap(&args, shards, *tenant).await,
        Command::SkegSubjectErase { args } => skeg_subject_erase(&args, shards, *tenant).await,
        Command::SkegTenantErase { args } => {
            skeg_tenant_erase(&args, shards, *tenant, tenant_backend).await
        }
        Command::SkegTenantDelete { args } => {
            skeg_tenant_delete(&args, shards, *tenant, tenant_backend).await
        }
        Command::SkegReclaim => skeg_reclaim(shards, *tenant, tenant_backend).await,
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

/// Accepts either the wire byte as a digit ("0".."5") or its display name
/// ("f32", "tq2", ...). Byte values come from [`QuantKind::from_wire`], the
/// single source of truth for the persisted-registry / RESP3-wire encoding.
fn parse_kind_arg(b: &Bytes) -> Result<u8, Frame> {
    let s = parse_utf8_arg(b, "kind")?;
    let s = s.to_ascii_lowercase();
    if let Ok(byte) = s.parse::<u8>()
        && QuantKind::from_wire(byte).is_some()
    {
        return Ok(byte);
    }
    for (byte, kind) in QuantKind::wire_kinds() {
        if kind.wire_name() == Some(s.as_str()) {
            return Ok(*byte);
        }
    }
    Err(Frame::Error(format!(
        "ERR unknown kind '{s}'; expected f32 | int8 | binary | tq1 | tq2 | tq4"
    )))
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
    // `scoped` came out of `scope_vindex_or_reject`, which refused the
    // separator in the raw name and then wrote the prefix itself, so this is
    // the pre-scoped door rather than the raw one.
    match shards
        .vindex_create_scoped(&scoped, dim, kind, backend)
        .await
    {
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

/// `SKEG.SUBJECT.ERASE prefix`. Erase every KV key of the calling tenant whose
/// app-key bytes start with `prefix` (a data subject within the tenant). Logical
/// delete; the value bytes leave the disk only on a later `SKEG.RECLAIM`.
/// Returns the number of keys erased.
async fn skeg_subject_erase(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'SKEG.SUBJECT.ERASE'".into());
    }
    // The anonymous tenant's keys are unscoped, so a subject prefix cannot be
    // told apart from any other key: refuse rather than risk a store-wide wipe.
    if tenant.is_zero() {
        return Frame::Error("ERR SKEG.SUBJECT.ERASE requires an authenticated tenant".into());
    }
    if args[0].is_empty() {
        return Frame::Error("ERR subject prefix must not be empty".into());
    }
    match shards
        .erase_prefix(tenant_u128(tenant), &args[0], DEFAULT_DURABILITY)
        .await
    {
        Ok(n) => Frame::Integer(n as i64),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.TENANT.ERASE tenant`. Admin only. Erase a whole named tenant: its
/// vindexes and every KV key. Returns `[vindexes_dropped, keys_erased]`.
async fn skeg_tenant_erase(
    args: &[Bytes],
    shards: &ShardSet,
    caller: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    let (_backend, target) = match admin_target(&args[0], caller, ctx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    match shards
        .erase_tenant(tenant_u128(target), DEFAULT_DURABILITY)
        .await
    {
        Ok((vindexes, keys)) => Frame::Array(vec![
            Frame::Integer(vindexes as i64),
            Frame::Integer(keys as i64),
        ]),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.TENANT.DELETE tenant`. Admin only. The full offboarding lifecycle:
/// erase the tenant's data, then remove its identity (logins + limits) so the
/// tenant ceases to exist. Returns `[vindexes, keys, logins_removed]`.
///
/// Data is erased *before* the identity is removed: if the identity went first
/// and the erase failed, the data would be orphaned under a tenant that can no
/// longer log in - the exact leak this command closes. This way a failure
/// leaves a still-valid tenant with erased data, which a retry finishes.
/// Erasure is logical (tombstones); run `SKEG.RECLAIM` to reclaim the bytes.
async fn skeg_tenant_delete(
    args: &[Bytes],
    shards: &ShardSet,
    caller: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    let (backend, target) = match admin_target(&args[0], caller, ctx) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let (vindexes, keys) = match shards
        .erase_tenant(tenant_u128(target), DEFAULT_DURABILITY)
        .await
    {
        Ok(counts) => counts,
        Err(e) => return shard_error(&e),
    };
    let logins = match backend.remove_tenant(target) {
        Ok(n) => n,
        Err(e) => {
            return Frame::Error(format!(
                "ERR data erased but identity removal failed: {e:?}"
            ));
        }
    };
    Frame::Array(vec![
        Frame::Integer(vindexes as i64),
        Frame::Integer(keys as i64),
        Frame::Integer(logins as i64),
    ])
}

/// `SKEG.RECLAIM`. Admin only. Physically reclaim every dead byte across the
/// store - the durable half of an erase. Store-wide (a segment interleaves all
/// tenants), so it is admin-gated, not tenant-facing. Returns bytes reclaimed.
async fn skeg_reclaim(
    shards: &ShardSet,
    caller: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    let Some(backend) = ctx else {
        return Frame::Error("ERR multi-tenant backend not configured".into());
    };
    if !backend.is_admin(caller) {
        return Frame::Error("ERR admin privileges required".into());
    }
    match shards.reclaim().await {
        Ok(freed) => Frame::Integer(freed as i64),
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
        if args[4].len() > MAX_PAYLOAD_BYTES {
            return Frame::Error(
                crate::admission::AdmissionError::RequestTooLarge {
                    what: "SKEG.VSET payload bytes",
                    limit: MAX_PAYLOAD_BYTES as u64,
                    got: args[4].len() as u64,
                }
                .wire_message(),
            );
        }
        Some(args[4].clone())
    } else {
        None
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    // Limits come from the pluggable backend; `None` (no backend / unlimited)
    // skips the corresponding enforcement entirely. `max_disk_bytes` covers
    // the payload blob the same way it already covers a KV `SET`
    // (`docs/adr-payload-transaction.md`, "Disk quota").
    let limit = tenant_backend.and_then(|b| b.limits(tenant).max_vectors);
    let disk_limit = tenant_backend.and_then(|b| b.limits(tenant).max_disk_bytes);
    match shards
        .vset_with_disk_limit(
            &scoped,
            id,
            vector,
            tenant_u128(tenant),
            limit,
            disk_limit,
            payload,
        )
        .await
    {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VMSET name (id vector payload)+` - bulk insert. Items fan out
/// concurrently so the durable payload-blob writes batch in the group committer.
/// Returns the number of items inserted.
/// Items one VMSET may carry.
///
/// A RESP array may declare `MAX_AGGREGATE_LEN` (1,048,576) elements and VMSET
/// reads them as triples, so one command could carry ~349,525 items - each
/// re-parsed into a `Vec<f32>`, which at 1024 dimensions is 1.4 GB materialised
/// before anything can refuse it. The batch is built first and fanned out
/// after, so memory admission never sees it: it is upstream of the governor,
/// not something the governor forgot to check.
///
/// 4096 because it is the number this engine already thinks in (`FLUSH_ROWS`)
/// and because the batches actually in use here are 128 to 200, so the cap
/// leaves twenty times the room anyone was using. A client with more to send
/// splits it, which is what it was already doing.
const MAX_VMSET_ITEMS: usize = 4096;

/// The longest one item's error may be in a VMSET reply.
///
/// The reply carries one per item, so what used to be a single error is now up
/// to `MAX_VMSET_ITEMS` of them. 256 bytes is longer than every error this
/// path produces (the longest names a vindex, two dimensions and a reason) and
/// the product is checked against `MAX_CONN_BUFFER` in the tests: a reply the
/// connection buffer cannot hold kills the connection instead of telling the
/// client which item failed.
const MAX_VMSET_ERROR_LEN: usize = 256;

/// Vector bytes one VMSET may carry, summed across its items.
///
/// The item cap alone does not bound the size: 4096 bulks of `MAX_BULK_LEN`
/// each is 2 TB on paper. The sum of the argument LENGTHS is knowable before a
/// single f32 is copied, which is the point - it bounds the copy, not the
/// frame. (The frame itself arriving at all is a parser-level question and is
/// recorded as one; it is not VMSET's to answer.)
///
/// 64 MiB: at 1024 dimensions the item cap already keeps a batch under 16 MiB,
/// so this only ever fires on dimensions far larger than anything measured -
/// which is exactly when it should.
const MAX_VMSET_BYTES: usize = 64 * 1024 * 1024;

/// The largest opaque payload blob one vector may carry: `SKEG.VSET`'s
/// optional `PAYLOAD` blob, and each `SKEG.VMSET` item's payload field.
/// Enforced at staging, before the write ever reaches a shard.
///
/// Without a ceiling here a stored payload has no bound at all, and
/// `SKEG.VSEARCH WITHPAYLOAD`'s reply bound has nothing honest to multiply
/// by `k`: what a read can answer starts at what a write may store. 1 MiB is
/// generous for an opaque blob (a chunk of source text, a small image
/// thumbnail, a JSON document) and small enough that `MAX_VSEARCH_K` hits at
/// this cap stay a bounded reply rather than an unbounded one.
const MAX_PAYLOAD_BYTES: usize = 1024 * 1024;

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
    // Counted BEFORE the triples are walked. Checking after would mean the
    // allocation this cap exists to prevent had already happened.
    let n_items = (args.len() - 1) / 3;
    if n_items > MAX_VMSET_ITEMS {
        return Frame::Error(
            crate::admission::AdmissionError::RequestTooLarge {
                what: "SKEG.VMSET items",
                limit: MAX_VMSET_ITEMS as u64,
                got: n_items as u64,
            }
            .wire_message(),
        );
    }
    // Lengths, not contents: no copy has happened yet, and this is what stops
    // one from happening.
    let vector_bytes: usize = args[1..].iter().skip(1).step_by(3).map(Bytes::len).sum();
    if vector_bytes > MAX_VMSET_BYTES {
        return Frame::Error(
            crate::admission::AdmissionError::RequestTooLarge {
                what: "SKEG.VMSET vector bytes",
                limit: MAX_VMSET_BYTES as u64,
                got: vector_bytes as u64,
            }
            .wire_message(),
        );
    }
    // Same shape as the vector-bytes check above, over the third field of
    // each triple: the largest single payload, not their sum - VSEARCH
    // WITHPAYLOAD's reply bound multiplies by this per-hit ceiling, so it is
    // one blob at a time that has to stay bounded, not the batch's total.
    if let Some(len) = args[1..].iter().skip(2).step_by(3).map(Bytes::len).max()
        && len > MAX_PAYLOAD_BYTES
    {
        return Frame::Error(
            crate::admission::AdmissionError::RequestTooLarge {
                what: "SKEG.VMSET payload bytes",
                limit: MAX_PAYLOAD_BYTES as u64,
                got: len as u64,
            }
            .wire_message(),
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
    let disk_limit = tenant_backend.and_then(|b| b.limits(tenant).max_disk_bytes);
    let results = shards
        .vmset_with_disk_limit(&scoped, items, tenant_u128(tenant), limit, disk_limit)
        .await;
    // An array of n, one per item, in request order: `+OK` or that item's
    // error. The count this used to return could not name the item that
    // failed, and the single error it returned instead said nothing about the
    // n-1 items that did not.
    Frame::Array(
        results
            .iter()
            .map(|r| match r {
                Ok(()) => Frame::ok(),
                Err(e) => match shard_error(e) {
                    Frame::Error(msg) => Frame::Error(cap_item_error(&msg)),
                    other => other,
                },
            })
            .collect(),
    )
}

/// One item's error, capped.
///
/// The reply is now n errors rather than one, so its size is the item cap
/// times this - and a reply the connection buffer cannot hold is a connection
/// that dies rather than a client that learns which item failed. Cut on a
/// character boundary: an error message can carry a vindex name, and names are
/// UTF-8.
fn cap_item_error(msg: &str) -> String {
    if msg.len() <= MAX_VMSET_ERROR_LEN {
        return msg.to_owned();
    }
    let mut end = MAX_VMSET_ERROR_LEN;
    while end > 0 && !msg.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &msg[..end])
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

/// `SKEG.VGET name id`: the stored f32 vector, little-endian bytes, or Null.
/// The read twin of VSET: what went in comes back bit-exact, so a client
/// never re-embeds a document whose vector the index already holds.
async fn skeg_vget(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    if args.len() != 2 {
        return Frame::Error("ERR wrong number of arguments for 'SKEG.VGET'; want name id".into());
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
    match shards.vget(&scoped, id).await {
        Ok(Some(v)) => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            Frame::Bulk(bytes.into())
        }
        Ok(None) => Frame::Null,
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VINDEX.RESHARD name`: train the router, move every row to its
/// semantic owner. Returns the number of rows moved.
/// `SKEG.VINDEX.SHARDS <index>` - per-shard LSM state. LIST sums across
/// shards, and thresholds are written per shard: reading the sum against a
/// per-shard threshold is how a real fix aimed at the wrong number.
async fn skeg_vindex_shards(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, name) {
        Ok(s) => s,
        Err(f) => return f,
    };
    match shards.vindex_per_shard(&scoped).await {
        Ok(lines) => Frame::Bulk(Bytes::from(lines.join("\n") + "\n")),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.HEALTH <index>` - is maintenance keeping up? Separate from CHECK,
/// which certifies integrity: an index can be perfectly intact and losing
/// recall because runs are piling up faster than they merge.
async fn skeg_health(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, name) {
        Ok(s) => s,
        Err(f) => return f,
    };
    match shards.health(&scoped).await {
        Ok(lines) => Frame::Bulk(Bytes::from(lines.join("\n") + "\n")),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VOWNER <index> <id> [id...]` - the shard holding each id, as a flat
/// array of integers (replica, when one exists, follows as a second entry;
/// -1 means none). Diagnostic: lets a benchmark report the PLACEMENT its
/// numbers were taken under.
async fn skeg_vowner(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, name) {
        Ok(s) => s,
        Err(f) => return f,
    };
    let mut ids = Vec::with_capacity(args.len() - 1);
    for a in &args[1..] {
        match std::str::from_utf8(a)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
        {
            Some(v) => ids.push(v),
            None => return Frame::Error("ERR id must be an unsigned integer".into()),
        }
    }
    match shards.owners_of(&scoped, &ids).await {
        Ok(rows) => Frame::Array(
            rows.into_iter()
                .flat_map(|(p, r)| {
                    [
                        Frame::Integer(i64::from(p)),
                        Frame::Integer(r.map_or(-1, i64::from)),
                    ]
                })
                .collect(),
        ),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.CHECK <index>` - the operator's fsck. Returns one line per problem
/// found, or a single `OK` line when the index is healthy. Read-only.
async fn skeg_check(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, name) {
        Ok(s) => s,
        Err(f) => return f,
    };
    match shards.check(&scoped).await {
        Ok(problems) if problems.is_empty() => Frame::Bulk(Bytes::from("OK\n")),
        Ok(problems) => Frame::Bulk(Bytes::from(problems.join("\n") + "\n")),
        Err(e) => shard_error(&e),
    }
}

async fn skeg_vindex_reshard(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.reshard(&scoped, 0.25, 15, tenant_u128(tenant)).await {
        Ok(moved) => Frame::Integer(moved as i64),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VINDEX.OVERLAP name [tau]`: targeted boundary replication.
async fn skeg_vindex_overlap(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let tau = match args.get(1) {
        Some(b) => match std::str::from_utf8(b)
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            Some(v) if v > 0.0 => v,
            _ => return Frame::Error("ERR tau must be a positive number".into()),
        },
        None => 0.05,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.overlap(&scoped, tau, tenant_u128(tenant)).await {
        Ok(n) => Frame::Integer(n as i64),
        Err(e) => shard_error(&e),
    }
}

/// `SKEG.VGRAPH name [count] [shard]`: text lines `n <id> <degree>` then
/// `e <from> <to>` - a one-hop sample of one shard's base graph, sized for a
/// force-directed view (default 120 seeds, shard 0).
async fn skeg_vgraph(args: &[Bytes], shards: &ShardSet, tenant: TenantId) -> Frame {
    let raw_name = match parse_utf8_arg(&args[0], "name") {
        Ok(s) => s,
        Err(e) => return e,
    };
    let count = match args.get(1) {
        Some(b) => match parse_u64_arg(b, "count") {
            Ok(v) => (v as usize).clamp(1, 2048),
            Err(e) => return e,
        },
        None => 120,
    };
    let shard = match args.get(2) {
        Some(b) => match parse_u64_arg(b, "shard") {
            Ok(v) => v as usize,
            Err(e) => return e,
        },
        None => 0,
    };
    let scoped = match scope_vindex_or_reject(tenant, raw_name) {
        Ok(s) => s,
        Err(e) => return e,
    };
    match shards.graph_sample(&scoped, shard, count).await {
        Ok((nodes, edges)) => {
            let mut body = String::new();
            for (id, deg) in nodes {
                body.push_str(&format!("n {id} {deg}\n"));
            }
            for (a, b) in edges {
                body.push_str(&format!("e {a} {b}\n"));
            }
            Frame::Bulk(Bytes::from(body))
        }
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
            for row in rows {
                let (name, dim, kind, backend, n_vectors) =
                    (row.name, row.dim, row.kind, row.backend, row.n_vectors);
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
                let kind_label = match QuantKind::from_wire(kind).and_then(|k| k.wire_name()) {
                    Some(name) => name,
                    None => {
                        return Frame::Error(format!("ERR unexpected kind byte {kind} from shard"));
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
                // `resident` is not decoration: every count on this line is
                // summed over the resident shards only, so anything less than
                // all of them means the numbers are a partial reading.
                body.push_str(&format!(
                    "name={visible_name} dim={dim} kind={kind_label} backend={backend_label} \
                     n_vectors={n_vectors} delta={} runs={} run_rows={} tombs={} base={} \
                     resident={}/{}\n",
                    row.delta,
                    row.runs,
                    row.run_rows,
                    row.tombs,
                    row.base,
                    row.shards_resident,
                    row.shards_total,
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

/// `SKEG.STATS`: the cache summary line, this process's own cost, then the
/// whole telemetry dump.
///
/// Takes no budget any more. The governor and the ingress class publish
/// themselves to the telemetry registry the dump reads, so this reply and a
/// `/metrics` scrape carry the same series by construction rather than
/// because two blocks of formatting were kept in step by hand.
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
            // Self-reported process cost, Prometheus-standard names: the
            // engine says what it uses instead of every operator deriving it
            // from ps. CPU is cumulative; consumers take window deltas.
            // Descriptors: one per vlog segment and per vindex segment file,
            // so an operator needs to see headroom BEFORE "too many open
            // files" turns into a failed open.
            // The memory budget and the ingress class used to be assembled
            // by hand, right here, which is why `/metrics` - the surface an
            // operator actually scrapes - could not see either of them: the
            // two numbers that say whether the server is about to start
            // refusing were visible only to whoever typed a Redis command.
            // They report themselves now, through the same dump both
            // surfaces read, so parity is a property of the code rather than
            // of somebody remembering to copy a block.
            let (fd_soft, _fd_hard) = skeg_platform::fd_limit();
            let fd_open = skeg_platform::open_fd_count();
            let process = format!(
                "# TYPE process_resident_memory_bytes gauge\n\
                 process_resident_memory_bytes {}\n\
                 # TYPE process_cpu_seconds_total counter\n\
                 process_cpu_seconds_total {:.3}\n\
                 # TYPE process_max_fds gauge\n\
                 process_max_fds {fd_soft}\n{}",
                skeg_platform::rss_bytes(),
                skeg_platform::cpu_seconds(),
                fd_open.map_or(String::new(), |n| format!(
                    "# TYPE process_open_fds gauge\nprocess_open_fds {n}\n"
                )),
            );
            let body = format!(
                "{cache_line}\n\n{process}\n{}",
                skeg_telemetry::stats::dump_text()
            );
            Frame::Bulk(Bytes::from(body))
        }
        Err(e) => shard_error(&e),
    }
}

/// The RESP3 error line for a shard error.
///
/// The first word of a RESP error IS the code - `ERR`, `WRONGTYPE`,
/// `LOADING` - and that is what a client routes on. Prefixing `ERR` in front
/// of a condition the caller should RETRY turns it into a generic failure it
/// should not, which is the whole difference between backpressure and an
/// error.
///
/// This used to be decided by looking at the FIRST FOURTEEN BYTES of a string
/// a shard had written, against a table of known code words. Two things were
/// wrong with that beyond the obvious: a refusal that forgot to write the
/// word lost its classification silently (the quota did, for its whole life),
/// and the native handler had no equivalent - a prefix is not something an
/// error code byte can be derived from. Now the refusal arrives typed and
/// composes its own line, and the native handler derives its byte from the
/// same classification.
///
/// Exhaustive: a variant added to `ShardError` does not compile until
/// somebody says which code word it carries.
fn shard_error(e: &crate::shard::ShardError) -> Frame {
    warn!("shard error: {e}");
    match e {
        crate::shard::ShardError::Admission(a) => Frame::Error(a.wire_message()),
        // A full VSEARCH pool is an admission refusal that predates the
        // enum, so it is mapped to its classification here rather than
        // classified here: the retryable bit is still decided in one place.
        crate::shard::ShardError::Busy => {
            Frame::Error(crate::admission::AdmissionError::Busy.wire_message())
        }
        crate::shard::ShardError::InvalidRequest(msg) => {
            crate::admission::debug_assert_not_a_smuggled_refusal(msg);
            Frame::Error(format!("ERR {msg}"))
        }
        crate::shard::ShardError::Unavailable => Frame::Error(format!("ERR {e}")),
        crate::shard::ShardError::Storage(msg) => {
            crate::admission::debug_assert_not_a_smuggled_refusal(msg);
            Frame::Error(format!("ERR {e}"))
        }
    }
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

/// One connection's side of a KV read's admission: the budget the reply has
/// to fit inside, and what the rest of the connection already holds.
///
/// `GET`/`MGET` are the one command shape whose reply size is knowable from
/// the STORE but not from the request (`reply_upper_bound` returns `None` for
/// them, and the only request-derived bound - `MAX_BULK_LEN` per key - refuses
/// ordinary four-key requests under any realistic class). The size is
/// knowable, though: it is in the index, one lookup per key and no read. So
/// these two go the other way round from every other command - they ask the
/// store how big the answer is, reserve THAT, and only then fetch it.
pub(crate) struct ReadAdmission<'a> {
    budget: &'a mut ConnectionBudget,
    /// Buffer-capacity bytes this connection already holds, which the
    /// reservation sits on top of. The same figure `flush_reply` is later
    /// given as `other_capacity`, so the charge taken here is the one the
    /// flush finds already paid.
    held: usize,
}

impl<'a> ReadAdmission<'a> {
    pub(crate) fn new(budget: &'a mut ConnectionBudget, held: usize) -> Self {
        Self { budget, held }
    }
}

/// Why a KV read did not happen. WIRE-NEUTRAL on purpose: both listeners run
/// the same preflight and each renders the answer in its own protocol, so the
/// classification cannot drift between them by being written down twice.
pub(crate) enum ReadRefusal {
    /// The server refused the reply before it was built.
    Admission(crate::admission::AdmissionError),
    /// The store could not answer.
    Shard(crate::shard::ShardError),
}

/// The refusal for a read whose size arithmetic does not fit a `usize`.
///
/// Unreachable with `u32` record sizes short of 2^32 keys in one request, and
/// typed anyway: an admission decision that can only be taken by overflowing
/// is an admission decision nobody took.
fn read_sum_overflow(keys: usize) -> ReadRefusal {
    skeg_telemetry::tick_counter(skeg_telemetry::Counter::KvReadRefused);
    ReadRefusal::Admission(crate::admission::AdmissionError::RequestTooLarge {
        what: "summed KV reply bytes",
        limit: u64::MAX,
        got: keys as u64,
    })
}

/// A read refusal as a RESP3 error frame. The native listener renders the
/// same value as an `Err` frame with a code byte; neither invents its own
/// classification.
fn read_refusal_frame(refusal: &ReadRefusal) -> Frame {
    match refusal {
        ReadRefusal::Admission(a) => Frame::Error(a.wire_message()),
        ReadRefusal::Shard(e) => shard_error(e),
    }
}

/// The largest number of keys one `MGET` may name.
///
/// A key costs about nine bytes on the wire (`$1\r\nk\r\n`) and about
/// [`PREFLIGHT_BYTES_PER_KEY`] in the structures that answer it, so without a
/// ceiling the request side of that ratio is the cheap side - the same shape
/// `MAX_VMSET_ITEMS` already caps for writes, and the same typed refusal. It
/// also bounds the shard batch: `value_sizes` and `mget_bounded` are ONE
/// `ShardReq` each, so an unbounded key count is an unbounded amount of work
/// for one shard worker to do before it looks at anything else.
pub(crate) const MAX_MGET_KEYS: usize = 4096;

/// What one key of a KV read costs the server BEFORE a byte of any value is
/// read: a `ScopedKey` (a `Bytes` handle plus a tenant id), the `Bytes` clone
/// handed to the shard set, the per-shard bucket entry, the `u32` size that
/// comes back, and the `(index, key, bound)` triple of the bounded batch -
/// each with its own vector's growth slack.
///
/// Deliberately generous and deliberately a constant: it bounds a
/// RESERVATION, not an allocation, and under-charging here is exactly the
/// hole it exists to close.
const PREFLIGHT_BYTES_PER_KEY: usize = 256;

/// Take (or extend) this read's reservation. `grow_to` holds AT LEAST the
/// figure given, so calling it again with a larger one extends rather than
/// replaces.
fn reserve_read(
    admission: &mut Option<&mut ReadAdmission<'_>>,
    want: usize,
) -> Result<(), ReadRefusal> {
    let Some(admission) = admission.as_mut() else {
        return Ok(());
    };
    let Some(target) = admission.held.checked_add(want) else {
        return Err(read_sum_overflow(want));
    };
    match admission.budget.grow_to(target) {
        Ok(()) => Ok(()),
        Err(e) => {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::KvReadRefused);
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::IngressRefusedGrowth);
            Err(ReadRefusal::Admission(
                crate::admission::AdmissionError::from(e),
            ))
        }
    }
}

/// The reply's frame shape, sized before it exists: one node per key plus the
/// array that holds them, which is exactly what `frame_upper_bound` will later
/// measure on the real `Frame`.
fn reply_bound_for(sizes: &[u32]) -> Option<usize> {
    let mut want = FRAME_NODE_OVERHEAD;
    for &size in sizes {
        want = want.checked_add((size as usize).checked_add(FRAME_NODE_OVERHEAD)?)?;
    }
    Some(want)
}

/// Measure, reserve, fetch - in that order, which is the whole of B1.
///
/// Every value's length comes from the index (`value_sizes`: one hashmap
/// lookup per key, no segment touched, nothing allocated for a value); the
/// sum is taken with checked arithmetic and reserved against the connection
/// budget; only then is a byte of any value read, and each read is bounded by
/// the size that was reserved for it so a concurrent overwrite cannot make the
/// measurement stale (`mget_bounded` -> `VLog::get_bounded`, which checks the
/// same `IndexEntry` the `pread` allocates from).
///
/// Three things happen before the first per-key allocation, in this order: the
/// arity cap, the reservation for the preflight's OWN per-key structures, and
/// only then their construction. Reserving the reply and not the machinery
/// that produces it leaves a request whose cheapest part is the wire.
///
/// A measurement that goes stale is RETRIED once, not refused. The guard is
/// fail-closed by construction - a record past its bound is never read - but
/// failing closed on a legitimate four-byte `GET` because another client
/// happened to write that key is a denial of service the client cannot fix,
/// and it happened: 4 refusals in 4000 reads under a writer alternating 1 KiB
/// and 256 KiB, and with certainty for a key created inside the window, whose
/// measured size is 0. So: re-measure, re-reserve, fetch again. Only a value
/// that has moved AGAIN in the second window is refused, and then as a
/// retryable condition naming the value, because the request was never the
/// problem.
///
/// `Err` is the refusal to send back, and when it is returned nothing was
/// fetched that the connection had not already been granted.
pub(crate) async fn fetch_within_budget(
    keys: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    admission: Option<&mut ReadAdmission<'_>>,
) -> Result<Vec<Option<Bytes>>, ReadRefusal> {
    // Before ANY allocation proportional to the key count, including this
    // function's own.
    if keys.len() > MAX_MGET_KEYS {
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::KvReadRefused);
        return Err(ReadRefusal::Admission(
            crate::admission::AdmissionError::RequestTooLarge {
                what: "keys in one KV read",
                limit: MAX_MGET_KEYS as u64,
                got: keys.len() as u64,
            },
        ));
    }
    let mut admission = admission;
    let Some(preflight) = keys.len().checked_mul(PREFLIGHT_BYTES_PER_KEY) else {
        return Err(read_sum_overflow(keys.len()));
    };
    reserve_read(&mut admission, preflight)?;

    let scoped: Vec<ScopedKey> = keys.iter().map(|k| scope_key(tenant, k)).collect();
    let bytes: Vec<Bytes> = scoped.iter().map(|k| k.as_bytes().clone()).collect();
    let view = shards.tenant(tenant_u128(tenant));

    let mut sizes = match view.value_sizes(&bytes).await {
        Ok(sizes) => sizes,
        Err(e) => return Err(ReadRefusal::Shard(e)),
    };

    // Two attempts: the measurement, and one re-measurement if a value moved
    // under it. Not a loop until it works - that would let a client hold a
    // shard worker for as long as another client keeps writing.
    for attempt in 0..2 {
        let Some(reply) = reply_bound_for(&sizes) else {
            return Err(read_sum_overflow(keys.len()));
        };
        let Some(want) = preflight.checked_add(reply) else {
            return Err(read_sum_overflow(keys.len()));
        };
        reserve_read(&mut admission, want)?;

        let got = match view.mget_bounded(&bytes, &sizes).await {
            Ok(got) => got,
            Err(e) => return Err(ReadRefusal::Shard(e)),
        };

        let mut out = Vec::with_capacity(got.len());
        let mut moved = None;
        for (i, slot) in got.into_iter().enumerate() {
            match slot {
                BoundedGet::Missing => out.push(None),
                BoundedGet::Found(v) => out.push(Some(v)),
                // The window the bound exists to close: between the
                // measurement and the read, somebody rewrote this key with a
                // larger value. Nothing was read for it.
                BoundedGet::Oversize { record_bytes } => {
                    moved = Some((sizes.get(i).copied().unwrap_or(0), record_bytes));
                    break;
                }
            }
        }
        let Some((measured, found)) = moved else {
            return Ok(out);
        };
        if attempt == 0 {
            skeg_telemetry::tick_counter(skeg_telemetry::Counter::KvReadRemeasured);
            // Re-measure everything, not just the key that moved: the second
            // probe costs one lookup per key and no IO, and a second stale
            // entry elsewhere in the batch would only send us round again.
            sizes = match view.value_sizes(&bytes).await {
                Ok(sizes) => sizes,
                Err(e) => return Err(ReadRefusal::Shard(e)),
            };
            continue;
        }
        skeg_telemetry::tick_counter(skeg_telemetry::Counter::KvReadRefused);
        return Err(ReadRefusal::Admission(
            crate::admission::AdmissionError::ValueChangedUnderRead {
                measured: u64::from(measured),
                found: u64::from(found),
            },
        ));
    }
    // Unreachable: both arms of the loop above either return or `continue`,
    // and the second iteration cannot `continue`. Written as a refusal rather
    // than an `unreachable!` because a panic on the read path is a worse
    // answer than a retryable error, whatever a future edit does to the loop.
    Err(ReadRefusal::Admission(
        crate::admission::AdmissionError::Busy,
    ))
}

async fn kv_get(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
    admission: Option<&mut ReadAdmission<'_>>,
) -> Frame {
    if args.len() != 1 {
        return Frame::Error("ERR wrong number of arguments for 'GET'".into());
    }
    // BEFORE the size probe, not after: the probe reaches the index with the
    // scoped key, and an anonymous connection naming another tenant's scoped
    // key would learn from a refusal-vs-null whether that key exists. The
    // forgery check is what stops it, so it has to come first.
    if anon_key_collides_with_tenant(tenant, &args[0], ctx) {
        return anon_forgery_error();
    }
    match fetch_within_budget(args, shards, tenant, admission).await {
        Ok(mut values) => match values.pop().flatten() {
            Some(v) => Frame::Bulk(v),
            None => Frame::Null,
        },
        Err(refusal) => read_refusal_frame(&refusal),
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

/// `APPEND key value` => integer reply with the new value length. Creates the
/// key (like SET) when absent.
async fn kv_append(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
) -> Frame {
    if args.len() != 2 {
        return Frame::Error("ERR wrong number of arguments for 'APPEND'".into());
    }
    if anon_key_collides_with_tenant(tenant, &args[0], ctx) {
        return anon_forgery_error();
    }
    let k = scope_key(tenant, &args[0]);
    let disk_limit = ctx.and_then(|b| b.limits(tenant).max_disk_bytes);
    match shards
        .tenant(k.accounting_tenant())
        .with_disk_limit(disk_limit)
        .append(k.as_bytes(), &args[1], DEFAULT_DURABILITY)
        .await
    {
        Ok(len) => Frame::Integer(len as i64),
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
    // Counted from the index, not from the values. `EXISTS k1 ... kn` used
    // to fetch every value in full and then throw them away - the same
    // unbudgeted materialisation B1 is about, for an answer that is a single
    // integer. A record size is `padded_record_size`, which is never zero for
    // a live key (it covers the record header before it covers a byte of
    // value), so a zero here means absent and nothing else.
    let scoped: Vec<Bytes> = args
        .iter()
        .map(|k| scope_key(tenant, k).as_bytes().clone())
        .collect();
    match shards
        .tenant(tenant_u128(tenant))
        .value_sizes(&scoped)
        .await
    {
        Ok(sizes) => Frame::Integer(sizes.iter().filter(|s| **s > 0).count() as i64),
        Err(e) => shard_error(&e),
    }
}

async fn kv_mget(
    args: &[Bytes],
    shards: &ShardSet,
    tenant: TenantId,
    ctx: Option<&Arc<dyn TenantBackend>>,
    admission: Option<&mut ReadAdmission<'_>>,
) -> Frame {
    if args.is_empty() {
        return Frame::Error("ERR wrong number of arguments for 'MGET'".into());
    }
    // Every key, before any of them is looked up: see `kv_get`.
    for key in args {
        if anon_key_collides_with_tenant(tenant, key, ctx) {
            return anon_forgery_error();
        }
    }
    match fetch_within_budget(args, shards, tenant, admission).await {
        Ok(values) => Frame::Array(
            values
                .into_iter()
                .map(|v| v.map_or(Frame::Null, Frame::Bulk))
                .collect(),
        ),
        Err(refusal) => read_refusal_frame(&refusal),
    }
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
    // Scope every key up front (the owned keys back the borrows below), then
    // hand the whole set to `mset`, which writes one atomic batch per shard.
    let scoped: Vec<_> = args.chunks(2).map(|c| scope_key(tenant, &c[0])).collect();
    let pairs: Vec<(&[u8], &[u8])> = scoped
        .iter()
        .zip(args.chunks(2))
        .map(|(k, c)| (k.as_bytes().as_ref(), c[1].as_ref()))
        .collect();
    // Every key of one MSET is scoped under the SAME connection tenant, so
    // one lookup covers the whole batch - same disk quota, same source, as
    // a single SET's.
    let accounting_tenant = scoped
        .first()
        .map_or(tenant_u128(tenant), ScopedKey::accounting_tenant);
    let disk_limit = ctx.and_then(|b| b.limits(tenant).max_disk_bytes);
    match shards
        .mset_with_disk_limit(&pairs, DEFAULT_DURABILITY, accounting_tenant, disk_limit)
        .await
    {
        Ok(()) => Frame::ok(),
        Err(e) => shard_error(&e),
    }
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

    /// An idle or between-frames socket (`buffered == 0`) must not pin more
    /// than a few KiB of decoder capacity - the connection semaphore's
    /// default limit means hundreds of these can be idle at once.
    /// An ingress budget over a headroom the test dictates, for the call
    /// sites that need one and are not testing it.
    fn test_ingress() -> Arc<crate::ingress::IngressBudget> {
        #[derive(Debug)]
        struct Fixed(crate::memory::Headroom);
        impl crate::memory::MemorySource for Fixed {
            fn headroom(&self) -> crate::memory::Headroom {
                self.0
            }
        }
        let governor = Arc::new(
            crate::memory::MemoryGovernor::new(
                Arc::new(Fixed(crate::memory::Headroom::Known(64 << 20))),
                None,
                Some(0),
            )
            .expect("a governor"),
        );
        let budget = Arc::new(crate::ingress::IngressBudget::new(
            governor,
            None,
            None,
            None,
            u64::from(u32::MAX),
        ));
        // What `Server::with_ingress_budget` does for a real listener: a
        // budget nothing registered is a budget `SKEG.STATS` cannot see, and
        // that is the property under test.
        budget.register_metrics();
        budget
    }

    /// An ingress budget whose class is far too small for a maximum reply, so
    /// the overshoot path can be driven without a megabyte-sized fixture.
    fn tiny_ingress() -> Arc<crate::ingress::IngressBudget> {
        #[derive(Debug)]
        struct Fixed(crate::memory::Headroom);
        impl crate::memory::MemorySource for Fixed {
            fn headroom(&self) -> crate::memory::Headroom {
                self.0
            }
        }
        let governor = Arc::new(
            crate::memory::MemoryGovernor::new(
                Arc::new(Fixed(crate::memory::Headroom::Known(1 << 20))),
                None,
                Some(0),
            )
            .expect("a governor"),
        );
        Arc::new(crate::ingress::IngressBudget::new(
            governor,
            None,
            Some(crate::ingress::CHUNK_BYTES),
            None,
            u64::from(u32::MAX),
        ))
    }

    #[test]
    fn read_reserve_idle_is_small() {
        assert!(super::read_reserve(0) <= 4096);
    }

    /// Once a frame is mid-flight, reserve the large chunk so a pipelined
    /// burst still buffers many frames per syscall.
    #[test]
    fn read_reserve_mid_frame_is_large() {
        assert_eq!(super::read_reserve(1), 256 * 1024);
    }

    /// The per-connection ceiling tracks the two caps it is built from, not
    /// the old 512 MiB default - a silent bump of any constant must fail this
    /// test - and it admits a VMSET at the vector cap with payloads attached.
    #[test]
    fn max_conn_buffer_is_bounded_to_one_frame() {
        const { assert!(super::MAX_CONN_BUFFER <= 130 * 1024 * 1024) };
        const { assert!(super::MAX_CONN_BUFFER > super::MAX_VMSET_BYTES + 1024) };
    }

    /// A VMSET reply is now one line per item, so the worst case is every item
    /// failing with the longest error each. That has to fit in what a
    /// connection will hold, or the reply that says which item failed is the
    /// thing that kills the connection.
    #[test]
    fn a_vmset_reply_of_nothing_but_errors_still_fits_a_connection() {
        // `-` + the message + CRLF per item.
        const WORST: usize = super::MAX_VMSET_ITEMS * (1 + super::MAX_VMSET_ERROR_LEN + 4 + 2);
        const { assert!(WORST < super::MAX_CONN_BUFFER) };
        assert_eq!(
            super::cap_item_error(&"x".repeat(super::MAX_VMSET_ERROR_LEN + 50)).len(),
            super::MAX_VMSET_ERROR_LEN + 3,
            "a long error is cut to the cap plus the ellipsis"
        );
        assert_eq!(super::cap_item_error("ERR short"), "ERR short");
        // A multi-byte character straddling the cut must not be halved.
        let wide = format!("{}e\u{301}", "x".repeat(super::MAX_VMSET_ERROR_LEN - 1));
        let cut = super::cap_item_error(&wide);
        assert!(
            std::str::from_utf8(cut.as_bytes()).is_ok(),
            "the cut landed inside a character"
        );
    }

    #[test]
    fn trim_idle_releases_a_drained_burst_buffer() {
        let mut buf = BytesMut::with_capacity(256 * 1024);
        super::trim_idle(&mut buf);
        assert!(buf.capacity() <= 4096, "capacity {}", buf.capacity());
    }

    #[test]
    fn trim_idle_keeps_a_mid_frame_buffer() {
        let mut buf = BytesMut::with_capacity(256 * 1024);
        buf.extend_from_slice(b"*");
        super::trim_idle(&mut buf);
        assert!(buf.capacity() >= 256 * 1024);
        assert_eq!(&buf[..], b"*");
    }

    #[tokio::test]
    async fn a_vmset_over_the_cap_is_refused_before_its_vectors_are_parsed() {
        // A RESP array may declare 1,048,576 elements, and VMSET reads them as
        // triples, so ONE command can carry ~349,525 items - each re-parsed
        // into a Vec<f32>. At 1024 dimensions that is 1.4 GB materialised
        // before admission sees anything, because the batch is built first and
        // fanned out after. The governor cannot refuse what it is never shown.
        //
        // The vectors here are deliberately INVALID. That is the discriminating
        // part: if the cap were checked after parsing, the answer would be a
        // parse error, and the test would pass while the batch had already been
        // walked.
        let dir = tempfile::TempDir::new().unwrap();
        let shards = crate::shard::ShardSet::open(dir.path(), 1).unwrap();
        let mut args = vec![Bytes::from_static(b"v")];
        for id in 0..(MAX_VMSET_ITEMS + 1) {
            args.push(Bytes::from(id.to_string()));
            // Three bytes: not a multiple of four, so `parse_vector` refuses
            // it. The first version of this used "not a vector", which is
            // TWELVE bytes and parses cleanly as three floats - so the batch
            // was walked without complaint and the test proved nothing.
            args.push(Bytes::from_static(b"abc"));
            args.push(Bytes::new());
        }
        let Frame::Error(msg) = skeg_vmset(&args, &shards, TenantId::ZERO, None).await else {
            panic!("an oversized batch must be refused");
        };
        assert!(
            msg.contains(&MAX_VMSET_ITEMS.to_string()),
            "the refusal must name the cap: {msg}"
        );
        assert!(
            !msg.contains("multiple of 4"),
            "refused by parsing a vector, which means the batch was walked: {msg}"
        );
    }

    #[tokio::test]
    async fn a_vmset_under_the_item_cap_can_still_be_too_large() {
        // The item cap does not bound size: 4096 bulks of MAX_BULK_LEN each is
        // 2 TB on paper. Two items are enough to show the byte cap is a
        // separate gate, and they are well under the item cap so only the byte
        // one can refuse them.
        let dir = tempfile::TempDir::new().unwrap();
        let shards = crate::shard::ShardSet::open(dir.path(), 1).unwrap();
        let big = Bytes::from(vec![0u8; MAX_VMSET_BYTES / 2 + 8]);
        let args = vec![
            Bytes::from_static(b"v"),
            Bytes::from_static(b"0"),
            big.clone(),
            Bytes::new(),
            Bytes::from_static(b"1"),
            big,
            Bytes::new(),
        ];
        let Frame::Error(msg) = skeg_vmset(&args, &shards, TenantId::ZERO, None).await else {
            panic!("an oversized batch must be refused");
        };
        assert!(
            msg.contains("vector bytes"),
            "refused for some other reason: {msg}"
        );
    }

    #[tokio::test]
    async fn stats_reports_the_memory_budget() {
        // A ceiling that is enforced and not readable leaves an operator to
        // find out from the refusals. The container gate says so directly: it
        // asks whether the engine SEES the limit, and a store that survives
        // without reporting one survived without knowing.
        let dir = tempfile::TempDir::new().unwrap();
        let shards = crate::shard::ShardSet::open(dir.path(), 1).unwrap();
        let Frame::Bulk(body) = skeg_stats(&shards).await else {
            panic!("a bulk summary");
        };
        let text = String::from_utf8(body.to_vec()).unwrap();
        for line in [
            "skeg_memory_budget_state",
            "skeg_memory_reserved_bytes",
            "skeg_memory_reserve_bytes",
        ] {
            assert!(text.contains(line), "STATS says nothing about {line}");
        }
        // The state is named, not encoded as a number a reader has to decode.
        assert!(
            text.contains("state=\"known\"")
                || text.contains("state=\"unlimited\"")
                || text.contains("state=\"unknown\""),
            "the budget state is not one of the three: {text}"
        );
    }

    /// A slowloris gate reads ONE number out of a running server to decide
    /// whether a thousand idle connections cost megabytes or gigabytes. If
    /// STATS does not carry it, the gate has to infer the answer from the
    /// process RSS, which is every allocation in the engine at once.
    #[tokio::test]
    async fn stats_reports_the_ingress_budget() {
        let dir = tempfile::TempDir::new().unwrap();
        let shards = crate::shard::ShardSet::open(dir.path(), 1).unwrap();
        let ingress = test_ingress();
        // Held for the length of the assertions: the registry keeps the
        // source by Weak, so a budget nobody holds is a budget that has
        // correctly stopped reporting.
        let held = ingress.try_accept().expect("a floor");
        let Frame::Bulk(body) = skeg_stats(&shards).await else {
            panic!("a bulk summary");
        };
        let text = String::from_utf8(body.to_vec()).unwrap();
        for line in [
            "skeg_ingress_state",
            "skeg_ingress_cap_bytes",
            "skeg_ingress_held_bytes",
            "skeg_ingress_per_connection_max_bytes",
        ] {
            assert!(text.contains(line), "STATS says nothing about {line}");
        }
        // Three states, named, exactly as the governor's own gauge does it.
        assert!(
            text.contains("skeg_ingress_state{state=\"known\"}")
                || text.contains("skeg_ingress_state{state=\"default\"}")
                || text.contains("skeg_ingress_state{state=\"unreadable\"}"),
            "the ingress state is not one of the three: {text}"
        );
        // And the held figure is the live one, not a constant.
        assert!(
            text.contains(&format!(
                "skeg_ingress_held_bytes {}",
                crate::ingress::FLOOR_BYTES
            )),
            "the held figure did not follow the accepted connection: {text}"
        );
        drop(held);
    }

    /// The reply buffer is the other half of "one budget", and it was in no
    /// budget at all. A VMSET of 4096 failing items is about 1.03 MiB of
    /// reply, `BytesMut` never gives capacity back, and 1024 connections that
    /// each sent one such batch pin a gigabyte the governor cannot see.
    #[tokio::test]
    async fn a_reply_buffer_is_charged_and_given_back_after_the_flush() {
        let budget = test_ingress();
        let mut conn = budget.try_accept().expect("the floor");
        let mut out = BytesMut::with_capacity(4096);
        let mut sink: Vec<u8> = Vec::new();

        // The worst reply this server can build: one line per item, each the
        // longest error the item cap allows. `frame_upper_bound` sizes this
        // reservation from the `Frame` BEFORE `encode_frame` runs (R1), so it
        // is taken from the frame's own content - not from `out`'s allocated
        // capacity, which `BytesMut`'s growth doubles past what it needs and
        // so routinely overshoots. That reservation comfortably fits this
        // connection's allowance (`test_ingress`'s per-connection ceiling is
        // several times the ~1.1 MiB this reply needs), so it succeeds.
        let reply = Frame::Array(
            (0..MAX_VMSET_ITEMS)
                .map(|_| Frame::Error("E".repeat(MAX_VMSET_ERROR_LEN)))
                .collect(),
        );
        let peak_bound = super::frame_upper_bound(&reply);
        assert!(
            flush_reply(
                &mut sink,
                &mut out,
                &reply,
                skeg_resp3::ProtoVersion::Resp3,
                &mut conn,
                4096,
            )
            .await
        );
        assert!(
            sink.len() > 1_000_000,
            "the reply was not the large one: {}",
            sink.len()
        );
        // The MEMORY is given back unconditionally: `out` is trimmed right
        // after the write regardless of what the budget could release.
        assert!(
            out.capacity() <= 4096,
            "the reply buffer was not given back: {} bytes still allocated",
            out.capacity()
        );
        // The BUDGET'S bookkeeping is chunk-granular (`shrink_to` only pops a
        // whole `CHUNK_BYTES`-rounded reservation, never a fraction of one),
        // so a connection that genuinely needed one large chunk does not
        // shrink back to the literal floor the moment its buffer empties -
        // that granularity is `ConnectionBudget`'s, not R1's to change here.
        // What R1 owns is that the charge is bounded to what was reserved
        // for THIS reply and does not silently grow further.
        assert!(
            conn.held_bytes() >= crate::ingress::FLOOR_BYTES,
            "held less than the floor a connection always keeps: {}",
            conn.held_bytes()
        );
        let expected_peak_charge = crate::ingress::charge_for(4096usize.saturating_add(peak_bound));
        assert_eq!(
            conn.held_bytes(),
            expected_peak_charge,
            "held does not match what this reply's own bound ({peak_bound}) should have \
             charged, chunk-rounded"
        );
        // And it does not creep further on a second, tiny reply: the charge
        // taken for the big one is the peak, not a floor a later flush
        // silently raises again.
        let held_after_first = conn.held_bytes();
        assert!(
            flush_reply(
                &mut sink,
                &mut out,
                &Frame::ok(),
                skeg_resp3::ProtoVersion::Resp3,
                &mut conn,
                4096,
            )
            .await
        );
        assert!(
            conn.held_bytes() <= held_after_first,
            "a tiny reply must not grow the charge past the big reply's peak: {} > {held_after_first}",
            conn.held_bytes()
        );
    }

    /// A reply is the answer to work that has already committed. When the
    /// class cannot cover its buffer the answer still goes out - withdrawing
    /// it would make a client retry a VMSET that has been applied - so the
    /// overshoot is COUNTED rather than turned into an error.
    #[tokio::test]
    async fn a_reply_too_large_for_the_class_is_still_delivered_and_counted() {
        let budget = tiny_ingress();
        let mut conn = budget.try_accept().expect("the floor");
        let mut out = BytesMut::with_capacity(4096);
        let mut sink: Vec<u8> = Vec::new();
        let before = skeg_telemetry::counter_value(skeg_telemetry::Counter::IngressReplyOverBudget);

        let reply = Frame::Array(
            (0..MAX_VMSET_ITEMS)
                .map(|_| Frame::Error("E".repeat(MAX_VMSET_ERROR_LEN)))
                .collect(),
        );
        assert!(
            flush_reply(
                &mut sink,
                &mut out,
                &reply,
                skeg_resp3::ProtoVersion::Resp3,
                &mut conn,
                4096,
            )
            .await,
            "the reply must be written even when the class cannot cover it"
        );
        assert!(
            sink.len() > 1_000_000,
            "the reply was truncated: {}",
            sink.len()
        );
        assert!(
            skeg_telemetry::counter_value(skeg_telemetry::Counter::IngressReplyOverBudget) > before,
            "the budget was exceeded in silence"
        );
        assert!(
            out.capacity() <= 4096,
            "the buffer was not given back: {}",
            out.capacity()
        );
    }

    #[test]
    fn a_retryable_refusal_reaches_the_client_as_the_code() {
        // ERR in front of BACKPRESSURE tells a client not to retry something
        // it should retry. The refusal arrives typed now, so the code word is
        // derived from the classification rather than read off the front of a
        // string somebody remembered to write.
        let f = shard_error(&crate::shard::ShardError::Admission(
            crate::admission::AdmissionError::MemoryAtWrite(
                crate::memory::MemoryRejected::NoHeadroom {
                    reserved: 1,
                    requested: 2,
                    usable: 1,
                },
            ),
        ));
        let Frame::Error(s) = f else {
            panic!("an error frame");
        };
        assert!(s.starts_with("BACKPRESSURE "), "the code was buried: {s}");
    }

    /// The guard covers BOTH prose-carrying variants, not only `Storage`.
    ///
    /// Debug builds only: that is where the assertion exists, and a release
    /// test asserting a no-op would pass without proving anything.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a refusal reached the wire as prose")]
    fn a_code_word_typed_into_an_invalid_request_is_caught() {
        let _ = shard_error(&crate::shard::ShardError::InvalidRequest(
            "BACKPRESSURE this is a refusal wearing the wrong variant".to_owned(),
        ));
    }

    #[test]
    fn a_permanent_refusal_reaches_the_client_as_a_plain_error() {
        // The other half of the same rule, and the reason the classification
        // has to be one decision: a permanent refusal dressed as backpressure
        // is a client that loops.
        let f = shard_error(&crate::shard::ShardError::Admission(
            crate::admission::AdmissionError::QuotaExceeded {
                tenant: 3,
                limit: 10,
            },
        ));
        let Frame::Error(s) = f else {
            panic!("an error frame");
        };
        assert!(
            s.starts_with("ERR "),
            "a quota does not clear on a retry: {s}"
        );
        assert!(s.contains("quota exceeded"), "{s}");
    }

    #[test]
    fn an_ordinary_error_still_gets_the_generic_code() {
        let f = shard_error(&crate::shard::ShardError::Storage(
            "vindex 'x' not found".to_owned(),
        ));
        let Frame::Error(s) = f else {
            panic!("an error frame");
        };
        assert!(
            s.starts_with("ERR "),
            "an error with no code of its own must get the generic one: {s}"
        );
        assert!(s.contains("vindex 'x' not found"), "{s}");
    }
    use super::{
        AUTH_FAIL_MAX, Command, TenantId, auth_clear, auth_is_blocked, auth_record_failure,
        is_pipelineable, scope_vindex_or_reject,
    };
    use bytes::Bytes;

    #[test]
    fn auth_throttle_blocks_after_max_failures_and_clears_on_success() {
        // Use a fixed, otherwise-unused IP so the process-global map is not
        // perturbed by (or perturbing) other tests.
        let ip = std::net::IpAddr::from([203, 0, 113, 7]);
        auth_clear(ip);
        assert!(!auth_is_blocked(ip));
        for _ in 0..AUTH_FAIL_MAX {
            auth_record_failure(ip);
        }
        assert!(auth_is_blocked(ip), "IP must be blocked after MAX failures");
        auth_clear(ip);
        assert!(
            !auth_is_blocked(ip),
            "successful login must clear the block"
        );
    }

    /// Path-traversal / cross-tenant guard: the index name flows into a
    /// filesystem path (`vindex-<name>`) and must never carry `..`, a path
    /// separator, or the `::` scope marker.
    #[test]
    fn vindex_name_rejects_traversal_and_scope_escape() {
        let t = TenantId::ZERO;
        for bad in [
            "../x",
            "../../tmp/x",
            "a/b",
            "a\\b",
            "..",
            ".",
            "",
            "victimhex::idx",
            "x\0y",
            "a b",
            "name.with/slash",
        ] {
            assert!(
                scope_vindex_or_reject(t, bad).is_err(),
                "must reject {bad:?}"
            );
        }
        for ok in ["myindex", "idx_1", "docs-v2", "a.b.c", "A9_-."] {
            assert!(scope_vindex_or_reject(t, ok).is_ok(), "must accept {ok:?}");
        }
    }

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
            // A subject erase is the caller acting on its own data: a KV write,
            // not admin. The two store-wide / cross-tenant ops ARE admin - a
            // misclassification here would let a tenant erase or reclaim across
            // the whole store.
            (
                Command::SkegSubjectErase {
                    args: args(&["subj/"]),
                },
                CommandKind::KvWrite,
            ),
            (
                Command::SkegTenantErase { args: args(&["t"]) },
                CommandKind::Admin,
            ),
            (
                Command::SkegTenantDelete { args: args(&["t"]) },
                CommandKind::Admin,
            ),
            (Command::SkegReclaim, CommandKind::Admin),
            (Command::Ping(None), CommandKind::Meta),
        ];
        for (cmd, want) in cases {
            assert_eq!(command_kind(&cmd), want, "misclassified {cmd:?}");
        }
    }

    #[test]
    fn command_cost_erasure_charges_a_premium() {
        // These walk the whole keyspace / rewrite segments, so the admission
        // gate must see a cost far above the flat-1 default.
        assert_eq!(
            command_cost(&Command::SkegSubjectErase {
                args: args(&["s/"]),
            }),
            100
        );
        assert_eq!(
            command_cost(&Command::SkegTenantErase { args: args(&["t"]) }),
            100
        );
        assert_eq!(command_cost(&Command::SkegReclaim), 1000);
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
            None,
            None,
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
            None,
            None,
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
    async fn erasure_verbs_enforce_their_privilege_boundary() {
        let (_dir, shards) = fresh_shards().await;
        // A backend whose is_admin default is false: every caller is a plain
        // tenant, none an admin.
        let backend: std::sync::Arc<dyn TenantBackend> = std::sync::Arc::new(RbacBackend {
            seen: std::sync::Mutex::new(Vec::new()),
        });
        let mut state = ConnectionState::new(0);

        // RECLAIM from a non-admin tenant: refused.
        let mut tenant = tid_from_name("t");
        let f = dispatch_command(
            Command::SkegReclaim,
            &mut state,
            &mut tenant,
            &shards,
            Some(&backend),
            None,
            None,
        )
        .await;
        assert!(
            matches!(&f, Frame::Error(e) if e.contains("admin privileges required")),
            "RECLAIM must require admin, got {f:?}"
        );

        // TENANT.ERASE from a non-admin tenant: refused.
        let f = dispatch_command(
            Command::SkegTenantErase { args: args(&["t"]) },
            &mut state,
            &mut tenant,
            &shards,
            Some(&backend),
            None,
            None,
        )
        .await;
        assert!(
            matches!(&f, Frame::Error(e) if e.contains("admin privileges required")),
            "TENANT.ERASE must require admin, got {f:?}"
        );

        // TENANT.DELETE from a non-admin tenant: refused (before any erase).
        let f = dispatch_command(
            Command::SkegTenantDelete { args: args(&["t"]) },
            &mut state,
            &mut tenant,
            &shards,
            Some(&backend),
            None,
            None,
        )
        .await;
        assert!(
            matches!(&f, Frame::Error(e) if e.contains("admin privileges required")),
            "TENANT.DELETE must require admin, got {f:?}"
        );

        // SUBJECT.ERASE from the anonymous tenant (id 0): refused, because its
        // keys are unscoped and a prefix sweep cannot be confined to it.
        let mut anon = TenantId::ZERO;
        let f = dispatch_command(
            Command::SkegSubjectErase {
                args: args(&["subj/"]),
            },
            &mut state,
            &mut anon,
            &shards,
            Some(&backend),
            None,
            None,
        )
        .await;
        assert!(
            matches!(&f, Frame::Error(e) if e.contains("authenticated tenant")),
            "SUBJECT.ERASE must refuse the anonymous tenant, got {f:?}"
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
                None,
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

    #[tokio::test]
    async fn vindex_list_reports_turboquant_tier() {
        let (_dir, shards) = fresh_shards().await;
        let tenant = TenantId::ZERO;
        let created =
            skeg_vindex_create(&args(&["tq2", "64", "tq2", "disk"]), &shards, tenant).await;
        assert!(matches!(created, Frame::Simple(ref s) if s == "OK"));

        let listed = skeg_vindex_list(&shards, tenant).await;
        assert!(matches!(
            listed,
            Frame::Bulk(ref body) if std::str::from_utf8(body).unwrap().contains("kind=tq2")
        ));
    }

    #[tokio::test]
    async fn vindex_list_preserves_turboquant_tier_after_restart() {
        let dir = TempDir::new().unwrap();
        let tenant = TenantId::ZERO;
        {
            let shards = ShardSet::open(dir.path(), 1).unwrap();
            let created =
                skeg_vindex_create(&args(&["tq1", "64", "tq1", "disk"]), &shards, tenant).await;
            assert!(matches!(created, Frame::Simple(ref s) if s == "OK"));
        }

        let shards = ShardSet::open(dir.path(), 1).unwrap();
        let listed = skeg_vindex_list(&shards, tenant).await;
        assert!(matches!(
            listed,
            Frame::Bulk(ref body) if std::str::from_utf8(body).unwrap().contains("kind=tq1")
        ));
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

        let resp = kv_mget(
            &args(&["k1", "k2", "k3"]),
            &shards,
            TenantId::ZERO,
            None,
            None,
        )
        .await;
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
            let r = kv_get(&args(&[k]), &shards, TenantId::ZERO, None, None).await;
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
    async fn append_creates_then_accumulates_and_returns_length() {
        let (_dir, shards) = fresh_shards().await;
        // Absent key: APPEND creates it, returns its length.
        let r = kv_append(&args(&["doc"]), &shards, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Error(_)), "APPEND needs a value arg");

        let r = kv_append(&args(&["doc", "ab"]), &shards, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Integer(2)), "new length after create");
        let r = kv_append(&args(&["doc", "cde"]), &shards, TenantId::ZERO, None).await;
        assert!(matches!(r, Frame::Integer(5)), "new length after append");

        let g = kv_get(&args(&["doc"]), &shards, TenantId::ZERO, None, None).await;
        assert!(matches!(g, Frame::Bulk(ref b) if &b[..] == b"abcde"));
    }

    #[tokio::test]
    async fn incr_starts_at_one_for_missing_key() {
        let (_dir, shards) = fresh_shards().await;
        let resp = kv_incr_by(&args(&["counter"]), &shards, 1, TenantId::ZERO, None).await;
        assert!(matches!(resp, Frame::Integer(1)));
        // Stored as a UTF-8 integer, GET-readable.
        let g = kv_get(&args(&["counter"]), &shards, TenantId::ZERO, None, None).await;
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

        let r_alice = kv_get(&args(&["k"]), &shards, alice, None, None).await;
        assert!(matches!(r_alice, Frame::Bulk(ref b) if &b[..] == b"alice-value"));
        let r_bob = kv_get(&args(&["k"]), &shards, bob, None, None).await;
        assert!(matches!(r_bob, Frame::Bulk(ref b) if &b[..] == b"bob-value"));

        // And the anonymous (ZERO) view must not see either: ZERO writes
        // are unprefixed and the scoped writes carry a non-zero prefix.
        let r_anon = kv_get(&args(&["k"]), &shards, TenantId::ZERO, None, None).await;
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
        let r_bob = kv_get(&args(&["k"]), &shards, bob, None, None).await;
        assert!(matches!(r_bob, Frame::Bulk(ref b) if &b[..] == b"bv"));
        let r_alice = kv_get(&args(&["k"]), &shards, alice, None, None).await;
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
        let r = kv_get(&args(&["hits"]), &shards, alice, None, None).await;
        assert!(matches!(r, Frame::Bulk(ref b) if &b[..] == b"3"));
        let r = kv_get(&args(&["hits"]), &shards, bob, None, None).await;
        assert!(matches!(r, Frame::Bulk(ref b) if &b[..] == b"5"));
    }

    #[tokio::test]
    async fn vindex_create_refuses_a_name_carrying_the_scope_separator() {
        // The RESP3 door. `scope_vindex_or_reject` refuses `::` before it
        // scopes, so an anonymous connection cannot hand-write the prefix that
        // makes a key read as another tenant's.
        let (_dir, shards) = fresh_shards().await;
        let victim = tid_from_name("victim");
        let squat = format!("{victim}::x");

        let f = skeg_vindex_create(
            &args(&[&squat, "8", "f32", "disk"]),
            &shards,
            TenantId::ZERO,
        )
        .await;
        assert!(
            matches!(&f, Frame::Error(e) if e.contains("must not contain '::'")),
            "expected a refusal naming the separator, got {f:?}"
        );
        let rows = shards.vindex_list().await.unwrap();
        assert!(
            !rows.iter().any(|r| r.name == squat),
            "the refused name must not exist: {rows:?}"
        );
    }

    #[tokio::test]
    async fn tenant_zero_cannot_squat_the_name_another_tenant_would_use() {
        // Name squatting, the second half of the P1: even without the erase,
        // creating `<hex of B>::x` on tenant 0 took the map key B's own `x`
        // would occupy, and B's create then failed with "already exists". The
        // refusal has to leave B's create working.
        let (_dir, shards) = fresh_shards().await;
        let victim = tid_from_name("victim");

        let squatted = skeg_vindex_create(
            &args(&[&format!("{victim}::x"), "8", "f32", "disk"]),
            &shards,
            TenantId::ZERO,
        )
        .await;
        assert!(
            matches!(squatted, Frame::Error(_)),
            "tenant 0 must not reach tenant B's namespace"
        );

        let mine = skeg_vindex_create(&args(&["x", "8", "f32", "disk"]), &shards, victim).await;
        assert!(
            matches!(&mine, Frame::Simple(s) if s == "OK"),
            "the victim's own create must still succeed, got {mine:?}"
        );
    }

    #[tokio::test]
    async fn a_tenant_scoped_index_reopens_with_its_payloads_findable() {
        // `warm_payload_indexes` reads each recovered vindex's blobs with the
        // tenant it takes from the map key, then marks the index loaded either
        // way. Warmed under the wrong tenant it reads zero blobs and still
        // marks it loaded, so every filtered search for that tenant comes back
        // empty, in silence. Tenant 0 is the one case where the scoped and the
        // bare name coincide - which is exactly why this shape needs a test of
        // its own.
        let dir = TempDir::new().unwrap();
        let t = tid_from_name("warm");
        let q = vec_arg(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        {
            let shards = ShardSet::open(dir.path(), 1).unwrap();
            let created = skeg_vindex_create(&args(&["idx", "8", "f32", "disk"]), &shards, t).await;
            assert!(matches!(&created, Frame::Simple(s) if s == "OK"));
            for (id, who) in [("1", "user=bob"), ("2", "user=alice")] {
                let set = skeg_vset(
                    &[
                        Bytes::from_static(b"idx"),
                        Bytes::copy_from_slice(id.as_bytes()),
                        q.clone(),
                        Bytes::from_static(b"PAYLOAD"),
                        Bytes::copy_from_slice(who.as_bytes()),
                    ],
                    &shards,
                    t,
                    None,
                )
                .await;
                assert!(matches!(&set, Frame::Simple(s) if s == "OK"));
            }
            shards.write_snapshot_and_payload_indexes().await;
        }

        // Reopen: recovery warms the payload index inside the readiness
        // barrier, before this search runs.
        let shards = ShardSet::open(dir.path(), 1).unwrap();
        let resp = skeg_vsearch(
            &[
                Bytes::from_static(b"idx"),
                Bytes::from_static(b"10"),
                Bytes::from_static(b"0"),
                q,
                Bytes::from_static(b"FILTER"),
                Bytes::from_static(b"user = alice"),
            ],
            &shards,
            t,
        )
        .await;
        match resp {
            Frame::Array(items) => {
                assert_eq!(
                    items.len(),
                    2,
                    "one [id, score] pair for alice; an empty answer here is the \
                     silent-miss bug: {items:?}"
                );
                assert!(matches!(&items[0], Frame::Bulk(b) if &b[..] == b"2"));
            }
            other => panic!("expected one filtered hit, got {other:?}"),
        }
    }
}
