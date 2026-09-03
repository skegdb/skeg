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
/// connection semaphore. Eight connections, each holding a frame open, must
/// pin at most the CLASS budget between them - not eight times whatever one
/// connection may hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "opens in commit 5 (resp3: accept, grow, stall)"]
async fn n_connections_mid_frame_pin_at_most_the_budget() {
    let cap = 16 * CHUNK_BYTES;
    let ingress = budget(cap, Duration::from_millis(50));
    let (addr, _dir) = resp3_server(&ingress, 64).await;

    let mut conns = Vec::new();
    for _ in 0..8 {
        let mut s = TcpStream::connect(addr).await.expect("connect");
        s.write_all(&dribbled_command_head(8 << 20))
            .await
            .expect("head");
        // A megabyte of the declared bulk, which no per-connection allowance
        // in this budget can hold.
        let _ = s.write_all(&vec![b'x'; 1 << 20]).await;
        conns.push(s);
    }
    // Give the server time to read what it is willing to read.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let held = ingress.held_bytes();
    assert!(
        held > 8 * FLOOR_BYTES,
        "no connection grew at all, so nothing was proved: {held}"
    );
    assert!(
        held <= cap,
        "eight connections pinned {held} bytes against a class cap of {cap}"
    );
    assert!(
        ingress.governor().reserved_bytes() <= cap,
        "the governor's total ran past the class cap"
    );
    drop(conns);
    until("the budget to come back", Duration::from_secs(5), || {
        ingress.held_bytes() == 0
    })
    .await;
}

/// A budget that is never given back is a leak with a nicer name. The close
/// path has to return it - and the failpoint is what proves the close path was
/// reached at all, rather than the assertion holding because the connection
/// was never served.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "opens in commit 5 (resp3: accept, grow, stall)"]
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
#[ignore = "opens in commit 5 (resp3: accept, grow, stall)"]
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
#[ignore = "opens in commit 5 (resp3: accept, grow, stall)"]
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
#[ignore = "opens in commit 6 (native: the same budget and the same semaphore)"]
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
#[ignore = "opens in commit 6 (native: the same budget and the same semaphore)"]
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
#[ignore = "opens in commit 7 (SKEG.STATS reports the ingress budget)"]
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
