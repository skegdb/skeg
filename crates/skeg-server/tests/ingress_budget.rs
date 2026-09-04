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

// ------------------------------------------------- the reply side, reads (A1)

/// One RESP3 bulk string argument.
fn bulk(arg: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
    out.extend_from_slice(arg);
    out.extend_from_slice(b"\r\n");
}

/// One RESP3 array command from bulk-string arguments.
fn resp_array(args: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("*{}\r\n", args.len()).as_bytes());
    for a in args {
        bulk(a, &mut out);
    }
    out
}

fn f32_le(v: f32) -> [u8; 4] {
    v.to_le_bytes()
}

/// `SKEG.VINDEX.CREATE name dim 0` (flat, f32) as raw RESP3 bytes.
fn vindex_create_flat(name: &str, dim: usize) -> Vec<u8> {
    resp_array(&[
        b"SKEG.VINDEX.CREATE",
        name.as_bytes(),
        dim.to_string().as_bytes(),
        b"0",
    ])
}

/// `SKEG.VINDEX.CREATE name dim 1` (disk, f32) as raw RESP3 bytes.
/// `SKEG.VGRAPH` samples a Vamana graph, which only a disk-backed index
/// builds - "flat backend has no graph".
fn vindex_create_disk(name: &str, dim: usize) -> Vec<u8> {
    resp_array(&[
        b"SKEG.VINDEX.CREATE",
        name.as_bytes(),
        dim.to_string().as_bytes(),
        b"1",
    ])
}

/// `SKEG.VSET name id vector PAYLOAD blob` as raw RESP3 bytes.
fn vset_with_payload(name: &str, id: u64, dim: usize, payload_len: usize) -> Vec<u8> {
    let mut vector = Vec::with_capacity(dim * 4);
    for _ in 0..dim {
        vector.extend_from_slice(&f32_le(0.5));
    }
    let payload = vec![b'p'; payload_len];
    resp_array(&[
        b"SKEG.VSET",
        name.as_bytes(),
        id.to_string().as_bytes(),
        &vector,
        b"PAYLOAD",
        &payload,
    ])
}

/// `SKEG.VSEARCH name k l_search vector WITHPAYLOAD` as raw RESP3 bytes.
fn vsearch_withpayload(name: &str, k: usize, dim: usize) -> Vec<u8> {
    let mut vector = Vec::with_capacity(dim * 4);
    for _ in 0..dim {
        vector.extend_from_slice(&f32_le(0.5));
    }
    resp_array(&[
        b"SKEG.VSEARCH",
        name.as_bytes(),
        k.to_string().as_bytes(),
        b"0",
        &vector,
        b"WITHPAYLOAD",
    ])
}

/// Read one whole RESP3 reply (one top-level frame) off `s`, with a deadline.
/// Good enough for a short `+OK`/`-ERR .../:` line; not a general RESP parser.
async fn read_one_reply(s: &mut TcpStream, within: Duration) -> String {
    let mut buf = Vec::new();
    let mut scratch = [0u8; 4096];
    loop {
        let n = tokio::time::timeout(within, s.read(&mut scratch))
            .await
            .expect("a reply must arrive")
            .expect("read");
        assert!(n > 0, "the server closed with no reply");
        buf.extend_from_slice(&scratch[..n]);
        if buf.ends_with(b"\r\n") {
            return String::from_utf8_lossy(&buf).into_owned();
        }
    }
}

/// audit/19 A1: `SKEG.VSET ... PAYLOAD <blob>` over `MAX_PAYLOAD_BYTES` is
/// refused before it ever reaches the shard - a typed `RequestTooLarge`, not
/// a silent unbounded store. Without this cap `SKEG.VSEARCH WITHPAYLOAD`'s
/// reply bound would have nothing honest to multiply `k` by.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_vset_payload_over_the_cap_is_refused_by_name() {
    let ingress = budget(64 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 8).await;
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&vindex_create_flat("payload-cap", 4))
        .await
        .expect("create");
    let _ = read_one_reply(&mut s, Duration::from_secs(5)).await;

    // One byte over the cap: the boundary is the point, not a wildly
    // oversized request that might be refused for some other reason.
    s.write_all(&vset_with_payload("payload-cap", 1, 4, 1024 * 1024 + 1))
        .await
        .expect("vset");
    let text = read_one_reply(&mut s, Duration::from_secs(10)).await;
    assert!(
        text.starts_with("-ERR "),
        "a payload over the cap is a permanent refusal, not backpressure: {text:?}"
    );
    assert!(
        text.contains("payload") && text.contains("1048576"),
        "the refusal must name what was too large and by how much: {text:?}"
    );

    // And it never touched the shard: the id it tried to write is absent.
    let mut check = TcpStream::connect(addr).await.expect("connect");
    check
        .write_all(&resp_array(&[b"SKEG.VGET", b"payload-cap", b"1"]))
        .await
        .expect("vget");
    let text = read_one_reply(&mut check, Duration::from_secs(5)).await;
    assert!(
        text.starts_with("_") || text.starts_with("$-1"),
        "the refused write must not have reached the shard: {text:?}"
    );
}

/// audit/19 A1, exit criterion: a `SKEG.VSEARCH k WITHPAYLOAD` whose worst
/// case (`k` hits, each up to `MAX_PAYLOAD_BYTES`) does not fit this
/// connection's allowance is refused BEFORE the shard call - proven against
/// an index name that does NOT exist, so a real shard call would answer
/// "not found" and a pre-dispatch admission refusal answers something else.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_max_vsearch_withpayload_reply_too_large_is_refused_before_the_shard_call() {
    // k=4096 WITHPAYLOAD needs roughly 4096 * 1 MiB - far past any
    // per-connection allowance a small class can grant.
    let ingress = budget(8 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 8).await;
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&vsearch_withpayload("does-not-exist", 4096, 4))
        .await
        .expect("vsearch");
    let text = read_one_reply(&mut s, Duration::from_secs(10)).await;
    assert!(
        text.starts_with("-ERR ") || text.starts_with("-BACKPRESSURE "),
        "an oversized reply must be refused before dispatch: {text:?}"
    );
    assert!(
        !text.to_lowercase().contains("not found") && !text.to_lowercase().contains("vindex"),
        "this refusal must come from admission, not from the shard resolving \
         (and refusing) a vindex name - which would prove the shard WAS \
         called: {text:?}"
    );
}

/// audit/19 A1, the reproduced hazard: pipelined `SKEG.VSEARCH k WITHPAYLOAD`
/// reads on ONE connection, sent without reading their replies. Before this
/// fix `reply_upper_bound` returned `None` for a read, so `exec_pipelined`
/// spawned every one of them unreserved and up to `PIPELINE_WINDOW` (128)
/// complete, payload-bearing replies could sit in `inflight` never charged -
/// the auditor measured 20,809,984 bytes of them under a 4 MiB class. Now
/// each pipelined read's bound is summed into `reply_reserved` before it is
/// allowed to run, so a class that cannot afford them all refuses the
/// requests it cannot afford instead of quietly completing and holding
/// every one.
///
/// `k=1` here, not the auditor's `k=50`: this test's bound is the WORST
/// CASE (`MAX_PAYLOAD_BYTES` per hit, since the reservation cannot know a
/// real stored payload is smaller), so even `k=1 WITHPAYLOAD` already
/// reserves past a megabyte - the point is proving the PIPELINE WINDOW's
/// running sum is enforced (several small, real reservations exhausting a
/// class together), not reproducing the auditor's exact numbers.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pipelined_vsearch_withpayload_reads_are_reserved_not_left_uncharged() {
    const DIM: usize = 8;
    const K: usize = 1;
    const BURST: usize = 20;

    // Sized to admit a handful of k=1 WITHPAYLOAD reservations
    // (MAX_PAYLOAD_BYTES-bounded, chunk-rounded to a bit over 2.5 MiB each)
    // but not all 20 of them summed - the class the auditor used was
    // 4 MiB for k=50; this is a different point on the same curve, chosen
    // so the test does not depend on exactly how many of BURST get through,
    // only that some do and some do not.
    let ingress = budget(64 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 8).await;

    let mut setup = TcpStream::connect(addr).await.expect("connect");
    setup
        .write_all(&vindex_create_flat("pipeline-idx", DIM))
        .await
        .expect("create");
    let _ = read_one_reply(&mut setup, Duration::from_secs(5)).await;
    setup
        .write_all(&vset_with_payload("pipeline-idx", 0, DIM, 256))
        .await
        .expect("vset");
    let text = read_one_reply(&mut setup, Duration::from_secs(5)).await;
    assert!(text.starts_with('+'), "setup vset must succeed: {text:?}");
    drop(setup);

    // The burst: BURST pipelined VSEARCH k=K WITHPAYLOAD requests, written
    // back to back on one connection with nothing read from it yet - the
    // exact shape the auditor reproduced.
    let mut conn = TcpStream::connect(addr).await.expect("connect");
    let request = vsearch_withpayload("pipeline-idx", K, DIM);
    for _ in 0..BURST {
        conn.write_all(&request).await.expect("vsearch");
    }

    // Drain every reply the pipeline window and its backlog produce, timing
    // each read so a connection that goes silent (rather than closing or
    // refusing by name) is a hang here, not a false pass.
    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut buf = Vec::new();
    let mut scratch = [0u8; 65536];
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if Instant::now() >= deadline {
            break;
        }
        let n = match tokio::time::timeout(Duration::from_secs(5), conn.read(&mut scratch)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => n,
            Ok(Err(_)) => break,
            Err(_) => break, // no more replies arriving within the window
        };
        buf.extend_from_slice(&scratch[..n]);
        // Count top-level replies by their leading byte, draining what has
        // fully arrived: `*` (array, a VSEARCH hit list) or `-` (an error).
        while !buf.is_empty() {
            let Some(end) = find_reply_end(&buf) else {
                break;
            };
            match buf[0] {
                b'*' => ok += 1,
                b'-' => refused += 1,
                other => panic!("unexpected reply lead byte {other:?}: {buf:?}"),
            }
            buf.drain(..end);
        }
    }

    assert!(
        ok + refused > 0,
        "no replies arrived at all: nothing was proved"
    );
    assert!(
        ok > 0,
        "a class sized for a handful of these must admit at least one: \
         {ok} completed, {refused} refused"
    );
    // The property under test: SOME requests were refused rather than the
    // server completing and holding all BURST of them (which is what
    // `held_bytes` staying at the class's per-connection floor while a
    // multi-megabyte backlog of completed replies sat uncharged looked
    // like before this fix).
    assert!(
        refused > 0,
        "a class sized for only a handful of a {BURST}-request pipelined \
         burst of k={K} WITHPAYLOAD reads must refuse some of them - {ok} \
         completed and 0 were refused, which is the unreserved-pipeline \
         shape audit 19 A1 reproduced"
    );

    until(
        "the connection to settle back near its floor",
        Duration::from_secs(10),
        || ingress.held_bytes() <= ingress.per_connection_max(),
    )
    .await;
}

/// Best-effort: the byte length of one complete top-level RESP3 reply at the
/// front of `buf`, if it has fully arrived. Handles exactly the two shapes
/// this test's replies take (`-...\r\n` and `*n\r\n` arrays of
/// `$len\r\nbytes\r\n` / `,double\r\n` / `_\r\n` elements), not the whole
/// grammar.
fn find_reply_end(buf: &[u8]) -> Option<usize> {
    fn line_end(buf: &[u8], from: usize) -> Option<usize> {
        buf[from..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|p| from + p + 2)
    }
    match buf.first()? {
        b'-' | b':' => line_end(buf, 0),
        b'$' => {
            let header_end = line_end(buf, 0)?;
            let len: usize = std::str::from_utf8(&buf[1..header_end - 2])
                .ok()?
                .parse()
                .ok()?;
            let end = header_end + len + 2;
            (buf.len() >= end).then_some(end)
        }
        b'*' => {
            let header_end = line_end(buf, 0)?;
            let n: usize = std::str::from_utf8(&buf[1..header_end - 2])
                .ok()?
                .parse()
                .ok()?;
            let mut pos = header_end;
            for _ in 0..n {
                match *buf.get(pos)? {
                    b'$' => {
                        let h = line_end(buf, pos)?;
                        let len: usize = std::str::from_utf8(&buf[pos + 1..h - 2])
                            .ok()?
                            .parse()
                            .ok()?;
                        pos = h + len + 2;
                        if buf.len() < pos {
                            return None;
                        }
                    }
                    b'_' => pos = line_end(buf, pos)?,
                    b',' | b':' => pos = line_end(buf, pos)?,
                    _ => return None,
                }
            }
            Some(pos)
        }
        _ => None,
    }
}

// ------------------------------------------------- the reply side, reads (A2)

/// `SKEG.VGRAPH name count` as raw RESP3 bytes.
fn vgraph_cmd(name: &str, count: usize) -> Vec<u8> {
    resp_array(&[
        b"SKEG.VGRAPH",
        name.as_bytes(),
        count.to_string().as_bytes(),
    ])
}

/// audit/19 A2: `SkegVgraph` is pipelineable (`is_pipelineable`) but round 1
/// left its bound `None`, the same shape A1 closed for `SkegVget`/
/// `SkegVsearch` with no structural reason `VGRAPH` should differ.
/// `count` clamps server-side to `[1, 2048]`; a typical Vamana out-degree
/// (~64) at `count=2048` puts one reply on the order of several MiB, and
/// `PIPELINE_WINDOW` (128) of those completed and uncharged on one
/// connection is the hundreds-of-MiB shape P0-A names for `SKEG.VMSET`.
///
/// `BURST` pipelined `SKEG.VGRAPH count=2048` reads on one connection, sent
/// without reading their replies, under a class sized for a handful: some
/// must be admitted (the class is not THAT small) and some must be refused
/// before they run (the class cannot hold all of `BURST`) - the same "some,
/// not all, not none" shape the VSEARCH pipeline test proves.
///
/// # Why the failpoint (audit 20, B7)
///
/// This test used to depend on the kernel. The connection loop drains its
/// pipeline whenever the decoder runs out of bytes, so whether several
/// reservations were ever outstanding AT ONCE was decided by whether the
/// server happened to drain between two of the client's writes. Under a
/// parallel build it did: all eight requests were admitted one at a time,
/// `refused` was 0 and the test failed - six times green in isolation, once
/// red under load, which is exactly the evidence a release gate must not
/// produce.
///
/// `HoldPipelineDrain` replaces that race with an event the test owns. While
/// it is armed the pipeline is not drained at a dry buffer, so every one of
/// `BURST` is submitted - and charged, or refused - before any of them is
/// emitted, however the kernel split the writes. The client then half-closes;
/// the server's read returns 0, leaves the loop, and the close path's drain
/// (which the failpoint does not guard) writes every reply in order. The
/// failpoint's `fired` flag is asserted, so a renamed or unreached point is a
/// failure rather than a test that quietly proves nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pipelined_vgraph_count_2048_reads_are_reserved_not_left_uncharged() {
    const DIM: usize = 8;
    const COUNT: usize = 2048;
    const BURST: usize = 8;

    // One count=2048 VGRAPH reservation (VGRAPH_MAX_OUT_DEGREE=64 edges per
    // node, VGRAPH_LINE_BYTES=48, chunk-rounded) is on the order of 12-13
    // MiB charged. Sized for roughly half of BURST to fit.
    let ingress = budget(560 * CHUNK_BYTES, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 8).await;
    let fp_key = key(addr);

    let mut setup = TcpStream::connect(addr).await.expect("connect");
    setup
        .write_all(&vindex_create_disk("graph-idx", DIM))
        .await
        .expect("create");
    let _ = read_one_reply(&mut setup, Duration::from_secs(5)).await;
    setup
        .write_all(&vset_with_payload("graph-idx", 0, DIM, 0))
        .await
        .expect("vset");
    let text = read_one_reply(&mut setup, Duration::from_secs(5)).await;
    assert!(text.starts_with('+'), "setup vset must succeed: {text:?}");
    drop(setup);

    // Armed after the setup connection is gone, so the setup traffic runs on
    // the ordinary path, and keyed on this listener's port so no other test in
    // this binary can see it.
    arm_ingress_at(IngressFailpoint::HoldPipelineDrain, &fp_key);

    let mut conn = TcpStream::connect(addr).await.expect("connect");
    let request = vgraph_cmd("graph-idx", COUNT);
    for _ in 0..BURST {
        conn.write_all(&request).await.expect("vgraph");
        // Deliberately spaced, and this is what makes the test STRICTER
        // rather than flakier: without the hold, a gap this size guarantees
        // the server drains between two writes and never holds more than one
        // reservation, which is the passing-for-the-wrong-reason state the
        // old version fell into by accident. With the hold, the outcome does
        // not depend on the gap at all.
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // The barrier: nothing is emitted until the peer stops writing, so all
    // BURST reservations are taken against the class together.
    conn.shutdown().await.expect("half-close");

    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut buf = Vec::new();
    let mut scratch = [0u8; 65536];
    loop {
        let n = match tokio::time::timeout(Duration::from_secs(20), conn.read(&mut scratch)).await {
            Ok(Ok(0)) => break,
            Ok(Ok(n)) => n,
            Ok(Err(e)) => panic!("read error before EOF: {e}"),
            Err(_) => panic!("the server neither answered nor closed within 20s"),
        };
        buf.extend_from_slice(&scratch[..n]);
        while !buf.is_empty() {
            let Some(end) = find_reply_end(&buf) else {
                break;
            };
            match buf[0] {
                b'$' => ok += 1,
                b'-' => refused += 1,
                other => panic!("unexpected reply lead byte {other:?}: {buf:?}"),
            }
            buf.drain(..end);
        }
    }

    disarm_ingress_at(IngressFailpoint::HoldPipelineDrain, &fp_key);
    assert!(
        fired_ingress_at(IngressFailpoint::HoldPipelineDrain, &fp_key),
        "the pipeline was never held, so nothing here was under test"
    );

    assert_eq!(
        ok + refused,
        BURST,
        "every request must be answered exactly once: {ok} completed, \
         {refused} refused"
    );
    assert!(
        ok > 0,
        "a class sized for a handful of these must admit at least one: \
         {ok} completed, {refused} refused"
    );
    assert!(
        refused > 0,
        "a class sized for only a handful of a {BURST}-request pipelined \
         burst of count={COUNT} VGRAPH reads must refuse some of them - \
         {ok} completed and 0 were refused, which is the unreserved-pipeline \
         shape audit 19 A2 named"
    );

    until(
        "the connection to settle back near its floor",
        Duration::from_secs(10),
        || ingress.held_bytes() <= ingress.per_connection_max(),
    )
    .await;
}
