//! P0.4: what the network is allowed to pin, over real sockets.
//!
//! The budget object has its own unit tests in `src/ingress.rs`. These are the
//! ones that cross an accept loop, because that is where the hazard lives: a
//! connection's buffer grows from bytes a peer chose to send, on a task the
//! accept loop spawned, and every previous ceiling in this server was
//! per-connection and therefore multiplied by the connection semaphore.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use skeg_server::Server;
use skeg_server::failpoint::{
    IngressFailpoint, arm_ingress_at, disarm_ingress_at, fired_ingress_at,
};
use skeg_server::ingress::{CHUNK_BYTES, FLOOR_BYTES, IngressBudget, PARSE_FACTOR};
use skeg_server::memory::{Headroom, MemoryGovernor, MemorySource};

// ---------------------------------------------------------------- fixtures

/// `Counter::IngressReplyOverBudget` is a process-wide static, and this
/// binary runs its `#[tokio::test]` functions concurrently by default (one
/// process, many threads) - so three tests below that each assert a DELTA
/// on it would otherwise measure each other's ticks instead of their own.
/// The same pattern as `vmset_fanout.rs`'s `ONE_AT_A_TIME`: take turns
/// rather than assert on a counter nothing else is touching, which nothing
/// here can promise on its own.
static REPLY_OVER_BUDGET_COUNTER_TESTS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Debug)]
struct Fixed(Headroom);

impl MemorySource for Fixed {
    fn headroom(&self) -> Headroom {
        self.0
    }
}

/// A budget whose class cap is exactly `cap` bytes, over headroom eight times
/// that so the governor is never the thing refusing.
fn budget(cap: u64, stall: Duration) -> Arc<IngressBudget> {
    let governor = Arc::new(
        MemoryGovernor::new(Arc::new(Fixed(Headroom::Known(cap * 8))), None, Some(0))
            .expect("a governor over a fixed headroom"),
    );
    Arc::new(IngressBudget::new(
        governor,
        None,
        Some(cap),
        Some(stall),
        u64::from(u32::MAX),
    ))
}

/// A running RESP3 server over the budget supplied, plus its address.
async fn resp3_server(
    ingress: &Arc<IngressBudget>,
    max_connections: usize,
) -> (std::net::SocketAddr, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind("127.0.0.1:0", dir.path())
        .await
        .expect("bind")
        .with_ingress_budget(Arc::clone(ingress))
        .with_max_connections(max_connections);
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run_resp3().await;
    });
    (addr, dir)
}

/// A running native-protocol server over the budget supplied.
async fn native_server(
    ingress: &Arc<IngressBudget>,
    max_connections: usize,
) -> (std::net::SocketAddr, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind("127.0.0.1:0", dir.path())
        .await
        .expect("bind")
        .with_ingress_budget(Arc::clone(ingress))
        .with_max_connections(max_connections);
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run().await;
    });
    (addr, dir)
}

/// The failpoint key: the listener's port. Every test binds port 0, so no two
/// share one - which is the isolation rule the keyed registry needs and that
/// a hand-chosen name cannot guarantee.
fn key(addr: std::net::SocketAddr) -> String {
    addr.port().to_string()
}

/// Poll `f` until it is true or the deadline passes.
async fn until(what: &str, deadline: Duration, mut f: impl FnMut() -> bool) {
    let end = Instant::now() + deadline;
    while Instant::now() < end {
        if f() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for {what}");
}

/// The head of a RESP array command whose LAST argument is a bulk of
/// `declared` bytes that the caller then dribbles. Nothing completes the
/// frame, which is exactly the shape a slow-drip attack takes.
fn dribbled_command_head(declared: usize) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"*4\r\n");
    for arg in [&b"SKEG.VSET"[..], b"idx", b"1"] {
        out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("${declared}\r\n").as_bytes());
    out
}

// ---------------------------------------------------------------- the tests

/// The multiplication P0.4 is about: a per-connection ceiling times the
/// connection semaphore. Sixteen connections, each holding a frame open and
/// pushing as hard as it can, must pin at most the CLASS budget between them -
/// not sixteen times whatever one connection may hold.
///
/// Sampled continuously rather than read once at the end, because the peak is
/// the number that matters and a connection refused for growing too far is
/// closed straight afterwards: a single reading taken a moment later would
/// pass on a server that had briefly pinned everything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn n_connections_mid_frame_pin_at_most_the_budget() {
    let cap = 16 * CHUNK_BYTES;
    let ingress = budget(cap, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 64).await;

    let peak = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sampler = {
        let ingress = Arc::clone(&ingress);
        let peak = Arc::clone(&peak);
        tokio::spawn(async move {
            for _ in 0..1500 {
                peak.fetch_max(ingress.held_bytes(), std::sync::atomic::Ordering::AcqRel);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    let mut writers = Vec::new();
    for _ in 0..16 {
        writers.push(tokio::spawn(async move {
            let mut s = TcpStream::connect(addr).await.expect("connect");
            if s.write_all(&dribbled_command_head(64 << 20)).await.is_err() {
                return;
            }
            // Dribble the declared bulk in small pieces for a while. The frame
            // never completes, which is the shape of the attack.
            let chunk = vec![b'x'; 32 * 1024];
            for _ in 0..400 {
                if s.write_all(&chunk).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }));
    }
    for w in writers {
        let _ = tokio::time::timeout(Duration::from_secs(20), w).await;
    }
    sampler.abort();

    let peak = peak.load(std::sync::atomic::Ordering::Acquire);
    assert!(
        peak > 16 * FLOOR_BYTES,
        "no connection grew at all, so nothing was proved: {peak}"
    );
    assert!(
        peak <= cap,
        "sixteen connections pinned {peak} bytes against a class cap of {cap}"
    );
    until("the budget to come back", Duration::from_secs(10), || {
        ingress.held_bytes() == 0
    })
    .await;
    assert_eq!(
        ingress.governor().reserved_bytes(),
        0,
        "the governor's total did not come back with the class share"
    );
}

/// A budget that is never given back is a leak with a nicer name. The close
/// path has to return it - and the failpoint is what proves the close path was
/// reached at all, rather than the assertion holding because the connection
/// was never served.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_budget_is_released_when_a_connection_closes() {
    let ingress = budget(16 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 64).await;
    let k = key(addr);
    arm_ingress_at(IngressFailpoint::ReleaseDeferredOnClose, &k);

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&dribbled_command_head(1 << 20))
        .await
        .expect("head");
    let _ = s.write_all(&vec![b'x'; 256 * 1024]).await;
    until("the connection to grow", Duration::from_secs(5), || {
        ingress.held_bytes() > FLOOR_BYTES
    })
    .await;

    drop(s);
    until("the budget to come back", Duration::from_secs(5), || {
        ingress.held_bytes() == 0
    })
    .await;
    let fired = fired_ingress_at(IngressFailpoint::ReleaseDeferredOnClose, &k);
    disarm_ingress_at(IngressFailpoint::ReleaseDeferredOnClose, &k);
    assert!(
        fired,
        "the close path never ran, so the budget coming back proves nothing"
    );
}

/// A frame bigger than one connection may ever hold is refused, by name, and
/// the bytes of it that already arrived are not kept while it is decided.
/// Buffering it first and refusing afterwards is the bug: the refusal costs
/// exactly what admitting it would have.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_frame_larger_than_the_connection_allowance_is_refused_not_buffered() {
    let cap = 8 * CHUNK_BYTES;
    let ingress = budget(cap, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 64).await;
    let allowance = ingress.per_connection_max();

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&dribbled_command_head(32 << 20))
        .await
        .expect("head");
    let writer = tokio::spawn(async move {
        // Push until the server stops taking it; the point is the refusal, not
        // the delivery.
        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..512 {
            if s.write_all(&chunk).await.is_err() {
                break;
            }
        }
        let mut reply = Vec::new();
        let _ = s.read_to_end(&mut reply).await;
        reply
    });

    let reply = tokio::time::timeout(Duration::from_secs(20), writer)
        .await
        .expect("the server must answer or close, not hang")
        .expect("writer task");
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with("-ERR ") || text.starts_with("-BACKPRESSURE "),
        "the refusal must lead with a code: {text:?}"
    );
    assert!(
        text.contains("ingress budget"),
        "the refusal must name what refused: {text:?}"
    );
    assert!(
        text.contains(&allowance.to_string()),
        "the refusal must carry the allowance: {text:?}"
    );
    until("the budget to come back", Duration::from_secs(5), || {
        ingress.held_bytes() == 0
    })
    .await;
}

/// A refused growth is not an immediate error: the connection stops reading,
/// which is free TCP backpressure, and only if the room has not come back
/// within the stall is the frame refused. Driven by the failpoint so the
/// sequence is exercised without arranging a real exhaustion, and asserted to
/// have fired so it cannot pass on a growth that was never refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_growth_stalls_then_refuses_the_frame_by_name() {
    let stall = Duration::from_millis(300);
    let ingress = budget(16 * CHUNK_BYTES, stall);
    let (addr, _dir) = resp3_server(&ingress, 64).await;
    let k = key(addr);
    arm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &k);

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&dribbled_command_head(4 << 20))
        .await
        .expect("head");
    let started = Instant::now();
    let mut writer = s;
    let reply = tokio::time::timeout(Duration::from_secs(20), async move {
        let chunk = vec![b'x'; 64 * 1024];
        for _ in 0..64 {
            if writer.write_all(&chunk).await.is_err() {
                break;
            }
        }
        let mut reply = Vec::new();
        let _ = writer.read_to_end(&mut reply).await;
        reply
    })
    .await
    .expect("the stall must end in a refusal, not a hang");
    let waited = started.elapsed();

    let fired = fired_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &k);
    disarm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &k);
    assert!(fired, "the growth was never refused, so nothing was proved");

    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with("-BACKPRESSURE "),
        "a stall that timed out is retryable and must say so: {text:?}"
    );
    assert!(
        waited >= stall,
        "the connection was refused without waiting out the stall: {waited:?}"
    );
}

/// The native protocol had no budget at all: no semaphore, an eager 64 KiB
/// buffer per connection, and a header that allocated its declared payload
/// before a byte of it arrived. It joins the same class, so a native peer and
/// a RESP3 peer draw on one total.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_connections_are_bounded_by_the_same_budget() {
    let cap = 16 * CHUNK_BYTES;
    let ingress = budget(cap, Duration::from_millis(50));
    let (addr, _dir) = native_server(&ingress, 64).await;

    let mut conns = Vec::new();
    for _ in 0..8 {
        let s = TcpStream::connect(addr).await.expect("connect");
        conns.push(s);
    }
    until(
        "the native connections to be charged",
        Duration::from_secs(5),
        || ingress.held_bytes() >= 8 * FLOOR_BYTES,
    )
    .await;
    assert!(
        ingress.held_bytes() <= cap,
        "eight idle native connections pinned {} against a cap of {cap}",
        ingress.held_bytes()
    );

    // And a declared payload larger than the connection allowance is refused
    // at the header - before the 24 bytes that declared it buy anything.
    let allowance = ingress.per_connection_max() / PARSE_FACTOR;
    let mut s = TcpStream::connect(addr).await.expect("connect");
    let mut header = [0u8; 24];
    header[0..2].copy_from_slice(&0x564Bu16.to_le_bytes());
    header[2] = 1; // version
    header[3] = 0x21; // Op::Set
    let declared = u32::try_from(allowance)
        .unwrap_or(u32::MAX)
        .saturating_mul(2);
    header[16..20].copy_from_slice(&declared.to_le_bytes());
    s.write_all(&header).await.expect("header");
    let mut reply = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), s.read_to_end(&mut reply))
        .await
        .expect("the server must answer or close, not hang")
        .expect("read");
    assert!(
        ingress.held_bytes() <= cap,
        "a declared length bought budget: {}",
        ingress.held_bytes()
    );
    drop(conns);
}

/// The native accept loop spawned a task per connection with nothing bounding
/// it. It now holds a permit for the connection's lifetime, exactly as the
/// RESP3 loop does.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn native_accept_is_bounded_by_the_connection_semaphore() {
    let ingress = budget(64 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = native_server(&ingress, 2).await;

    // PING, native: header only, no payload.
    let ping = |req_id: u64| {
        let mut h = [0u8; 24];
        h[0..2].copy_from_slice(&0x564Bu16.to_le_bytes());
        h[2] = 1;
        h[3] = 0x01; // Op::Ping
        h[8..16].copy_from_slice(&req_id.to_le_bytes());
        h
    };
    async fn answered(s: &mut TcpStream, within: Duration) -> bool {
        let mut buf = [0u8; 24];
        tokio::time::timeout(within, s.read_exact(&mut buf))
            .await
            .is_ok_and(|r| r.is_ok())
    }

    let mut held = Vec::new();
    for i in 0..2u64 {
        let mut s = TcpStream::connect(addr).await.expect("connect");
        s.write_all(&ping(i)).await.expect("ping");
        assert!(
            answered(&mut s, Duration::from_secs(5)).await,
            "connection {i} within the limit was not served"
        );
        held.push(s);
    }

    let mut third = TcpStream::connect(addr).await.expect("connect");
    third.write_all(&ping(99)).await.expect("ping");
    assert!(
        !answered(&mut third, Duration::from_millis(500)).await,
        "a third connection was served past a limit of two"
    );

    // Free a permit; the parked connection is served.
    held.pop();
    assert!(
        answered(&mut third, Duration::from_secs(10)).await,
        "the connection never got the permit that was freed"
    );
}

/// A thousand connections that say nothing. This is the shape the ceiling was
/// meant to stop, and the one no per-connection number ever bounded: each is
/// legitimate, and their sum is the whole machine.
///
/// The real binary, because the number that matters is the one an operator
/// reads out of `SKEG.STATS` on a server they started, not one a test computed
/// for itself.
#[test]
fn slowloris_one_thousand_idle_connections_hold_only_the_floor() {
    const IDLE: usize = 1000;
    // The test process needs a descriptor per connection of its own; the
    // default soft limit on macOS is 256, which would fail this for a reason
    // that has nothing to do with the server.
    let fds = skeg_platform::raise_fd_limit(u64::try_from(IDLE).unwrap_or(1024) + 256);
    assert!(
        fds >= IDLE as u64 + 64,
        "this machine will not give the test {IDLE} descriptors (got {fds}); \
         raise the hard limit with ulimit -n"
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skeg-resp3"));
    cmd.arg("--addr")
        .arg("127.0.0.1:0")
        .arg("--data-dir")
        .arg(dir.path())
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn skeg-resp3");
    let stdout = child.stdout.take().expect("stdout");
    let stderr = child.stderr.take().expect("stderr");
    let mut guard = KillOnDrop(child);

    let addr = wait_for_listening(stdout, stderr, &mut guard.0);
    let mut idle = Vec::with_capacity(IDLE);
    for i in 0..IDLE {
        match std::net::TcpStream::connect(addr) {
            Ok(s) => idle.push(s),
            Err(e) => panic!("connection {i} refused: {e}"),
        }
    }

    let held = read_ingress_held(addr);
    let floor_total = FLOOR_BYTES * (IDLE as u64 + 1);
    // Printed, because the number is the point: a gate that only says "under
    // the ceiling" cannot tell an operator what a connection actually costs.
    println!(
        "slowloris: {IDLE} idle connections hold {held} bytes, {} per connection",
        held / (IDLE as u64 + 1)
    );
    assert!(
        held >= FLOOR_BYTES * IDLE as u64 / 2,
        "the idle connections were not charged at all ({held}), so the number \
         proves nothing"
    );
    assert!(
        held <= floor_total * 2,
        "{IDLE} idle connections hold {held} bytes; the floor for them all is \
         {floor_total}"
    );
    drop(idle);
}

// ------------------------------------------------- real-binary plumbing

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Read the child's output until it says what it bound, with a deadline.
/// Both streams are drained on their own threads so a full pipe cannot wedge
/// the parent.
fn wait_for_listening(
    stdout: std::process::ChildStdout,
    stderr: std::process::ChildStderr,
    child: &mut Child,
) -> std::net::SocketAddr {
    let (tx, rx) = mpsc::channel::<String>();
    for mut reader in [
        Box::new(stdout) as Box<dyn Read + Send>,
        Box::new(stderr) as Box<dyn Read + Send>,
    ] {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0
                    || tx
                        .send(String::from_utf8_lossy(&buf[..n]).into_owned())
                        .is_err()
                {
                    break;
                }
            }
        });
    }
    drop(tx);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut combined = String::new();
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            panic!("the server exited with {status:?}; output: {combined}");
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => combined.push_str(&chunk),
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {}
        }
        if let Some(rest) = combined.split("listening on ").nth(1)
            && let Some(line) = rest.split_whitespace().next()
            && let Ok(addr) = line.trim().parse()
        {
            return addr;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for the listen address; output: {combined}"
        );
    }
}

/// Ask a running server what ingress currently holds.
fn read_ingress_held(addr: std::net::SocketAddr) -> u64 {
    use std::io::Write;
    let mut s = std::net::TcpStream::connect(addr).expect("stats connection");
    s.set_read_timeout(Some(Duration::from_secs(10)))
        .expect("timeout");
    s.write_all(b"*2\r\n$5\r\nHELLO\r\n$1\r\n3\r\n")
        .expect("hello");
    let mut scratch = [0u8; 8192];
    let _ = s.read(&mut scratch).expect("hello reply");
    s.write_all(b"*1\r\n$10\r\nSKEG.STATS\r\n").expect("stats");

    let mut body = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let n = s.read(&mut scratch).expect("stats reply");
        assert!(n > 0, "the server closed before answering SKEG.STATS");
        body.extend_from_slice(&scratch[..n]);
        let text = String::from_utf8_lossy(&body);
        if let Some(v) = text
            .lines()
            .find_map(|l| l.strip_prefix("skeg_ingress_held_bytes "))
            .and_then(|v| v.trim().parse::<u64>().ok())
        {
            return v;
        }
        assert!(
            Instant::now() < deadline,
            "SKEG.STATS never reported skeg_ingress_held_bytes: {text}"
        );
    }
}

// ------------------------------------------------- the reply side (A1)

/// A `SKEG.VMSET` whose every item fails against a missing index whose name is
/// as long as the server allows: 4096 reply lines, each at the item error cap,
/// from a request an order of magnitude smaller.
///
/// The long name is what makes the test about the REPLY. Each item's error is
/// "vindex '<255 chars>' not found", capped at 256 bytes, so the answer is
/// about 1 MiB while the request stays around 100 KiB - and the request is
/// already charged today.
fn vmset_of_failing_items(items: usize) -> Vec<u8> {
    let name = "n".repeat(255); // MAX_VINDEX_NAME_LEN
    let mut out = Vec::new();
    out.extend_from_slice(format!("*{}\r\n", 2 + items * 3).as_bytes());
    let bulk = |arg: &[u8], out: &mut Vec<u8>| {
        out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
        out.extend_from_slice(arg);
        out.extend_from_slice(b"\r\n");
    };
    bulk(b"SKEG.VMSET", &mut out);
    bulk(name.as_bytes(), &mut out);
    for id in 0..items {
        bulk(id.to_string().as_bytes(), &mut out);
        bulk(&1.0f32.to_le_bytes(), &mut out);
        bulk(b"", &mut out);
    }
    out
}

/// R1 (P0-A): a `SKEG.VMSET` whose worst-case reply does not fit this
/// connection's allowance is refused BEFORE it runs, not answered in full
/// and counted afterwards.
///
/// This is the behaviour A1/P0.4 explicitly declared out of scope
/// ("the one place the budget is knowingly exceeded"): the reply used to be
/// built - the mutation already committed - and only then charged, with an
/// overshoot counted rather than refused. `reply_upper_bound` in
/// `resp3_handler.rs` now computes this exact worst case
/// (`MAX_VMSET_ITEMS * (MAX_VMSET_ERROR_LEN + framing)`) from the request
/// alone and reserves it before `SKEG.VMSET`'s items are allowed to run at
/// all, so a class too small for the reply refuses the REQUEST - cheaply,
/// before a single item is attempted - rather than accepting it and
/// overshooting the budget on the way out.
///
/// The class here is sized so one connection's REQUEST fits its allowance
/// comfortably and request-plus-worst-case-reply does not, which is exactly
/// what makes this a test of the reply side and not the request side.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_max_vmset_reply_too_large_for_the_allowance_is_refused_before_it_runs() {
    let _turn = REPLY_OVER_BUDGET_COUNTER_TESTS.lock().await;
    let ingress = budget(16 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 64).await;
    let before = skeg_telemetry::counter_value(skeg_telemetry::Counter::IngressReplyOverBudget);

    let request = vmset_of_failing_items(4096);
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&request).await.expect("vmset");

    // A refusal is one short line, not 4096 of them: read whatever arrives
    // within the deadline and stop as soon as the line ends, rather than
    // waiting for 4096 markers that a refusal will never produce.
    let mut reply = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = tokio::time::timeout(Duration::from_secs(10), s.read(&mut buf))
            .await
            .expect("the server must answer or close, not hang")
            .expect("read");
        assert!(n > 0, "the server closed with no reply at all");
        reply.extend_from_slice(&buf[..n]);
        if reply.ends_with(b"\r\n") {
            break;
        }
    }
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with("-ERR ") || text.starts_with("-BACKPRESSURE "),
        "a pre-commit refusal must lead with a code, not the 4096-item array: {text:?}"
    );
    assert!(
        text.matches('\n').count() == 1,
        "this must be ONE short refusal line, not the reply array: {} lines",
        text.matches('\n').count()
    );

    // No overshoot to count: the reservation failed before any item ran, so
    // there was never a committed answer whose size the governor had to
    // accept after the fact.
    assert_eq!(
        skeg_telemetry::counter_value(skeg_telemetry::Counter::IngressReplyOverBudget),
        before,
        "a request refused before it ran must not tick the overshoot counter"
    );

    until(
        "the connection to hold only its floor",
        Duration::from_secs(10),
        || ingress.held_bytes() <= FLOOR_BYTES,
    )
    .await;
}

/// R1 (P0-A) exit criterion: N sockets, each sending the same max-size
/// `SKEG.VMSET` at once under a barrier, must never tick the overshoot
/// counter - not because the requests are refused (the class here is sized
/// to admit all of them), but because each connection's worst-case reply was
/// reserved before it ran and the actual reply never exceeds what was
/// reserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_max_vmset_replies_under_a_barrier_never_overshoot_the_budget() {
    let _turn = REPLY_OVER_BUDGET_COUNTER_TESTS.lock().await;
    const N: usize = 8;
    // Sized to hold N worst-case VMSET replies (~1.15 MiB each, charged at
    // PARSE_FACTOR and rounded up to a whole CHUNK_BYTES) plus each
    // connection's request buffer, with generous headroom - the point of
    // this test is concurrent admission succeeding cleanly, not admission
    // refusing under a class too small to hold them (that is the test
    // above). "Small" per the audit's own framing is relative to what N
    // legitimate worst-case replies cost, not to an arbitrary constant. The
    // margin is wide (32x a single reply's chunk-rounded charge per
    // connection) because this workspace's test suite runs several worktrees
    // and test binaries concurrently on one machine: a connection whose
    // request read gets scheduled in smaller, slower slices grows a larger
    // decoder buffer than it would alone, and that buffer shares this same
    // class - a tight margin here would make the assertion about scheduler
    // noise, not about the reservation.
    let ingress = budget(N as u64 * 32 * CHUNK_BYTES, Duration::from_millis(500));
    let (addr, _dir) = resp3_server(&ingress, 64).await;
    let before = skeg_telemetry::counter_value(skeg_telemetry::Counter::IngressReplyOverBudget);

    let request = std::sync::Arc::new(vmset_of_failing_items(4096));
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(N));
    let mut workers = Vec::new();
    for _ in 0..N {
        let request = std::sync::Arc::clone(&request);
        let barrier = std::sync::Arc::clone(&barrier);
        workers.push(tokio::spawn(async move {
            let mut s = TcpStream::connect(addr).await.expect("connect");
            // Every connection sends its max-size VMSET in the same instant,
            // which is the scenario the audit named: concurrent connections
            // each triggering a max VMSET error reply at once.
            barrier.wait().await;
            s.write_all(&request).await.expect("vmset");
            let mut seen = 0usize;
            let mut got = 0usize;
            let mut buf = vec![0u8; 64 * 1024];
            while seen < 4096 {
                let n = tokio::time::timeout(Duration::from_secs(30), s.read(&mut buf))
                    .await
                    .expect("the reply must arrive")
                    .expect("read");
                assert!(n > 0, "the server closed after {seen} reply lines");
                seen += buf[..n].iter().filter(|&&b| b == b'-').count();
                got += n;
            }
            got
        }));
    }

    let mut reply_lens = Vec::new();
    for w in workers {
        reply_lens.push(
            tokio::time::timeout(Duration::from_secs(30), w)
                .await
                .expect("a worker must finish, not hang")
                .expect("worker task"),
        );
    }

    assert_eq!(reply_lens.len(), N);
    assert!(
        reply_lens.iter().all(|&len| len > 1_000_000),
        "every connection must have received the full worst-case reply: {reply_lens:?}"
    );
    assert_eq!(
        skeg_telemetry::counter_value(skeg_telemetry::Counter::IngressReplyOverBudget),
        before,
        "N connections admitted concurrently, each answering its own \
         reserved worst case, must never overshoot: the reservation IS the \
         bound, not an afterthought counted once it is exceeded"
    );
}

/// End-to-end fairness under a class that is genuinely full.
///
/// The unit test of the per-connection allowance is arithmetic; this is the
/// socket version, and it is the one that says a saturated class still serves
/// the connections that are behaving.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_saturated_class_still_answers_ping_and_refuses_a_grower_by_name() {
    // A class driven this close to full can legitimately push even a small
    // reply's own `flush_reply` reservation (R1) past what little room is
    // left - `charge_for` has no granularity between "fits in the floor"
    // and "a whole CHUNK_BYTES", so a PONG whose OWN reservation needs a
    // few bytes more than the connection already holds asks for a full
    // chunk, same as a page's worth of reply would. That is a real,
    // COUNTED overshoot on `IngressReplyOverBudget` - not a bug in this
    // test's own scenario - and the two other tests in this file that
    // assert a delta of zero on that same process-wide counter take turns
    // with this one so its legitimate ticks are not mistaken for theirs.
    let _turn = REPLY_OVER_BUDGET_COUNTER_TESTS.lock().await;
    let cap = 16 * CHUNK_BYTES;
    let stall = Duration::from_millis(300);
    let ingress = budget(cap, stall);
    let (addr, _dir) = resp3_server(&ingress, 128).await;

    // A connection that exists before the class fills, and must go on being
    // served after it does.
    let mut early = TcpStream::connect(addr).await.expect("connect");
    early
        .write_all(b"*1\r\n$4\r\nPING\r\n")
        .await
        .expect("ping");
    let mut pong = [0u8; 7];
    tokio::time::timeout(Duration::from_secs(5), early.read_exact(&mut pong))
        .await
        .expect("the early connection must be served")
        .expect("read");

    // Holders: real connections, each buffering a partial frame and settling
    // at one growth chunk. Small on purpose - the read loop reserves the next
    // 256 KiB BEFORE it reads, so a connection holding a lot is one read-ahead
    // away from asking past its own allowance and being expelled for it, which
    // is correct behaviour and useless as a fixture.
    let body = vec![b'x'; 64 * 1024];
    let mut holders = Vec::new();
    for _ in 0..24 {
        if ingress.cap().bytes() - ingress.held_bytes() < 4 * CHUNK_BYTES {
            break;
        }
        let mut s = TcpStream::connect(addr).await.expect("connect");
        s.write_all(&dribbled_command_head(64 << 20))
            .await
            .expect("head");
        s.write_all(&body).await.expect("body");
        holders.push(s);
        // Let the server read it before deciding whether to add another.
        tokio::time::sleep(Duration::from_millis(60)).await;
    }
    let held = ingress.held_bytes();
    assert!(
        held > cap / 4,
        "the holders never took the class anywhere ({held} of {cap}), so \
         nothing below is a test of a full class"
    );

    // The last of the room, taken by the test itself in floor-sized steps, so
    // what is left is EXACTLY the state the assertions are about: too little
    // for any connection to grow, enough for one to be accepted. Sockets
    // cannot be aimed that precisely - a holder refused mid-flight closes and
    // gives everything back - and a test that could not aim would be asserting
    // about whatever the last holder happened to leave.
    let mut filler = Vec::new();
    while ingress.cap().bytes() - ingress.held_bytes() >= CHUNK_BYTES {
        match ingress.try_accept() {
            Ok(b) => filler.push(b),
            Err(_) => break,
        }
    }
    while ingress.cap().bytes() - ingress.held_bytes() < 8 * FLOOR_BYTES && filler.pop().is_some() {
    }
    let free = ingress.cap().bytes() - ingress.held_bytes();
    // Printed, because "the class was full" is the premise of everything
    // below and a reader should not have to take it on trust.
    println!(
        "saturated: {} holders, {held} of {cap} held by them, {free} free, \
         one growth chunk is {CHUNK_BYTES}",
        holders.len()
    );
    assert!(
        (8 * FLOOR_BYTES..CHUNK_BYTES).contains(&free),
        "the class is not in the state this test is about: {free} free"
    );

    // A brand new connection still gets its floor and its answer...
    let mut fresh = TcpStream::connect(addr).await.expect("connect");
    fresh
        .write_all(b"*1\r\n$4\r\nPING\r\n")
        .await
        .expect("ping");
    tokio::time::timeout(Duration::from_secs(5), fresh.read_exact(&mut pong))
        .await
        .expect("a full class must still accept and answer a small request")
        .expect("read");
    assert_eq!(&pong, b"+PONG\r\n");

    // ...and so does the one that was already there.
    early
        .write_all(b"*1\r\n$4\r\nPING\r\n")
        .await
        .expect("ping");
    tokio::time::timeout(Duration::from_secs(5), early.read_exact(&mut pong))
        .await
        .expect("a connection already served must not be starved by a full class")
        .expect("read");
    assert_eq!(&pong, b"+PONG\r\n");

    // But one more connection that wants to GROW is told to retry, after the
    // stall and not before it, and is not left hanging.
    let started = Instant::now();
    let mut grower = TcpStream::connect(addr).await.expect("connect");
    grower
        .write_all(&dribbled_command_head(64 << 20))
        .await
        .expect("head");
    // Enough to need ONE growth chunk and no more: the refusal has to come
    // from the class being full, not from this connection asking past its own
    // allowance - different refusals with different codes, and only the first
    // is retryable.
    let reply = tokio::time::timeout(Duration::from_secs(30), async move {
        let _ = grower.write_all(&vec![b'x'; 200 * 1024]).await;
        let mut reply = Vec::new();
        let _ = grower.read_to_end(&mut reply).await;
        reply
    })
    .await
    .expect("a full class must refuse, not hang");
    let waited = started.elapsed();
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with("-BACKPRESSURE "),
        "a full class is momentary and must say so: {text:?}"
    );
    assert!(
        waited >= stall,
        "refused without waiting out the stall: {waited:?}"
    );
    drop((holders, filler));
}
