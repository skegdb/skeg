//! B1 (audit 20, CWE-400): what a KV READ is allowed to allocate.
//!
//! `SKEG.VSET`/`SKEG.VMSET`/`SKEG.VSEARCH` reserve a bound taken from the
//! request before they run (P0-A, audit 19). `GET` and `MGET` did not: they
//! are not pipelineable, so no ONE connection can accumulate several of them,
//! and the reasoning stopped there. It should not have. A single
//! `MGET k1 k2 ... kn` names `n` keys in a few hundred bytes and each value
//! behind them is bounded only by `MAX_BULK_LEN` (64 MiB), so one small
//! request can ask the server to materialise gigabytes - and `N` connections
//! can each ask at the same time. The values were fetched into a `Vec<Frame>`
//! first and charged afterwards, which means the governor read the peak off an
//! allocation that had already happened.
//!
//! These tests are about the ORDER: the length comes from the index, the sum
//! is reserved, and only then is a byte of any value read.

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::BytesMut;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use skeg_proto::{ErrCode, Flags, FrameParser, Op, decode_err_response};
use skeg_server::Server;
use skeg_server::ingress::IngressBudget;
use skeg_server::memory::{Headroom, MemoryGovernor, MemorySource};
use skeg_telemetry::{Counter, counter_value};

// ---------------------------------------------------------------- fixtures

/// The counters these tests assert DELTAS on are process-wide statics and
/// this binary runs its `#[tokio::test]` functions concurrently, so two tests
/// reading the same counter would measure each other's ticks. Take turns.
static COUNTER_TESTS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Debug)]
struct Fixed(Headroom);

impl MemorySource for Fixed {
    fn headroom(&self) -> Headroom {
        self.0
    }
}

/// A budget whose class cap is exactly `cap` bytes, over headroom eight times
/// that so the governor is never the thing refusing.
fn budget(cap: u64) -> Arc<IngressBudget> {
    let governor = Arc::new(
        MemoryGovernor::new(Arc::new(Fixed(Headroom::Known(cap * 8))), None, Some(0))
            .expect("a governor over a fixed headroom"),
    );
    Arc::new(IngressBudget::new(
        governor,
        None,
        Some(cap),
        Some(Duration::from_millis(50)),
        u64::from(u32::MAX),
    ))
}

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

fn bulk(arg: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
    out.extend_from_slice(arg);
    out.extend_from_slice(b"\r\n");
}

fn resp_array(args: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("*{}\r\n", args.len()).as_bytes());
    for a in args {
        bulk(a, &mut out);
    }
    out
}

/// Read one complete top-level RESP3 reply, whatever its shape, by counting
/// what the leading byte promises. Bulk-aware, so a multi-megabyte `MGET`
/// answer is read to its end instead of being cut at the first `\r\n`.
async fn read_reply(s: &mut TcpStream, within: Duration) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut scratch = vec![0u8; 64 * 1024];
    let deadline = Instant::now() + within;
    loop {
        if let Some(end) = reply_end(&buf) {
            buf.truncate(end);
            return buf;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        assert!(!left.is_zero(), "no complete reply within {within:?}");
        let n = tokio::time::timeout(left, s.read(&mut scratch))
            .await
            .expect("a reply must arrive")
            .expect("read");
        assert!(n > 0, "the server closed with no reply");
        buf.extend_from_slice(&scratch[..n]);
    }
}

/// Byte length of one complete reply at the front of `buf`, if it has all
/// arrived. Covers the shapes these tests provoke: a simple string, an error,
/// an integer, a null, a bulk, and one array of those.
fn reply_end(buf: &[u8]) -> Option<usize> {
    fn line_end(buf: &[u8], from: usize) -> Option<usize> {
        buf.get(from..)?
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|p| from + p + 2)
    }
    fn one(buf: &[u8], at: usize) -> Option<usize> {
        match buf.get(at)? {
            b'+' | b'-' | b':' | b',' | b'_' | b'#' => line_end(buf, at),
            b'$' => {
                let head = line_end(buf, at)?;
                let len: i64 = std::str::from_utf8(&buf[at + 1..head - 2])
                    .ok()?
                    .parse()
                    .ok()?;
                if len < 0 {
                    return Some(head);
                }
                let end = head + usize::try_from(len).ok()? + 2;
                (buf.len() >= end).then_some(end)
            }
            b'*' => {
                let head = line_end(buf, at)?;
                let n: i64 = std::str::from_utf8(&buf[at + 1..head - 2])
                    .ok()?
                    .parse()
                    .ok()?;
                let mut pos = head;
                for _ in 0..n.max(0) {
                    pos = one(buf, pos)?;
                }
                Some(pos)
            }
            _ => None,
        }
    }
    one(buf, 0)
}

/// Fill the store with `n` keys, each holding `value_len` bytes, over a
/// connection of its own so the reads under test start from a clean budget.
async fn seed(addr: std::net::SocketAddr, n: usize, value_len: usize) -> Vec<Vec<u8>> {
    let mut s = TcpStream::connect(addr).await.expect("connect");
    let value = vec![b'v'; value_len];
    let mut keys = Vec::with_capacity(n);
    for i in 0..n {
        let key = format!("k{i:04}").into_bytes();
        s.write_all(&resp_array(&[b"SET", &key, &value]))
            .await
            .expect("set");
        let reply = read_reply(&mut s, Duration::from_secs(30)).await;
        assert!(
            reply.starts_with(b"+OK"),
            "seed SET {i} failed: {:?}",
            String::from_utf8_lossy(&reply)
        );
        keys.push(key);
    }
    keys
}

fn mget_command(keys: &[Vec<u8>]) -> Vec<u8> {
    let mut args: Vec<&[u8]> = Vec::with_capacity(keys.len() + 1);
    args.push(b"MGET");
    for k in keys {
        args.push(k.as_slice());
    }
    resp_array(&args)
}

/// The class most of these tests run under. 32 MiB gives a per-connection
/// allowance of 8 MiB (`cap / 4`), which admits a 256 KiB `SET` comfortably
/// and cannot admit sixty-four 256 KiB values in one `MGET` reply.
const CAP: u64 = 32 * 1024 * 1024;
/// Value size for the seeded keys.
const VALUE: usize = 256 * 1024;

// ------------------------------------------------------------- RESP3 reads

/// The B1 exit criterion: an `MGET` whose values do not fit the connection's
/// allowance is refused BEFORE the store is read, and the proof that nothing
/// was fetched is the counter of value bytes materialised by a read staying
/// at zero across the whole request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_huge_mget_is_refused_before_any_value_is_fetched() {
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(CAP);
    let (addr, _dir) = resp3_server(&ingress, 8).await;
    // 64 x 256 KiB = 16 MiB of values against an 8 MiB allowance.
    let keys = seed(addr, 64, VALUE).await;

    let before = counter_value(Counter::KvReadBytesFetched);
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&mget_command(&keys)).await.expect("mget");
    let reply = read_reply(&mut s, Duration::from_secs(30)).await;
    let text = String::from_utf8_lossy(&reply).into_owned();

    assert!(
        text.starts_with("-ERR ") || text.starts_with("-BACKPRESSURE "),
        "an MGET past the allowance must be refused by name, not answered: {:?}",
        &text[..text.len().min(200)]
    );
    assert_eq!(
        counter_value(Counter::KvReadBytesFetched) - before,
        0,
        "the refusal must precede the fetch: value bytes were materialised \
         for a request that was refused"
    );
}

/// The other half of the same criterion: an `MGET` that DOES fit is still
/// answered in full. A preflight that refuses everything is not a fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_mget_inside_the_allowance_is_answered_in_full() {
    // This one MATERIALISES a megabyte of values, so it moves
    // `KvReadBytesFetched` - the counter three tests below assert a zero
    // delta on. Taking the turn is what stops it being their flake (audit 25,
    // F3): the same shape B7 has just removed from `ingress_budget.rs`.
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(CAP);
    let (addr, _dir) = resp3_server(&ingress, 8).await;
    let keys = seed(addr, 4, VALUE).await;

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&mget_command(&keys)).await.expect("mget");
    let reply = read_reply(&mut s, Duration::from_secs(30)).await;
    assert!(
        reply.starts_with(b"*4\r\n"),
        "a four-key MGET inside the allowance must be answered: {:?}",
        String::from_utf8_lossy(&reply[..reply.len().min(120)])
    );
    assert!(
        reply.len() > 4 * VALUE,
        "the answer must carry every value: {} bytes",
        reply.len()
    );
}

/// `GET` of one value larger than the connection's allowance: the same order,
/// one key instead of many.
///
/// The value is grown with `APPEND` rather than written by one `SET`, because
/// a value large enough to be unreadable under a class is also too large to
/// arrive in a single frame under the same class - which is exactly how a
/// hostile client would build it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_get_over_the_allowance_is_refused_before_the_value_is_fetched() {
    const CHUNK: usize = 1024 * 1024;
    const CHUNKS: usize = 8;
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(CAP);
    let (addr, _dir) = resp3_server(&ingress, 8).await;

    let mut w = TcpStream::connect(addr).await.expect("connect");
    let chunk = vec![b'a'; CHUNK];
    for i in 0..CHUNKS {
        w.write_all(&resp_array(&[b"APPEND", b"big", &chunk]))
            .await
            .expect("append");
        let reply = read_reply(&mut w, Duration::from_secs(30)).await;
        assert!(
            reply.starts_with(b":"),
            "append {i} must succeed: {:?}",
            String::from_utf8_lossy(&reply)
        );
    }
    drop(w);

    let before = counter_value(Counter::KvReadBytesFetched);
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&resp_array(&[b"GET", b"big"]))
        .await
        .expect("get");
    let reply = read_reply(&mut s, Duration::from_secs(30)).await;
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with("-ERR ") || text.starts_with("-BACKPRESSURE "),
        "a GET past the allowance must be refused by name: {:?}",
        &text[..text.len().min(200)]
    );
    assert_eq!(
        counter_value(Counter::KvReadBytesFetched) - before,
        0,
        "the refusal must precede the fetch"
    );
}

/// The multiplication B1 names: not one big request, but N connections each
/// asking at the same time. Under a class that cannot hold all of them the
/// budget must REFUSE (typed, before the fetch) rather than let every reply be
/// built and then count the overshoot - which is what
/// `skeg_ingress_reply_over_budget_total` counts, and why it must stay at zero
/// here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn n_concurrent_mgets_never_overshoot_the_class() {
    // Half the connections ask for eight values (2 MiB of reply, ~4 MiB
    // charged, comfortably inside the 8 MiB allowance) and half ask for
    // thirty-two (8 MiB of reply, ~16 MiB charged, twice the allowance).
    // That makes BOTH outcomes structural rather than a race for the class:
    // four must be answered and four must be refused whatever the scheduler
    // does, so the refusal branch this test documents is actually taken -
    // audit 25, F4, the same demand B7 makes of its twin.
    const NARROW: usize = 4;
    const WIDE: usize = 4;
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(CAP);
    let (addr, _dir) = resp3_server(&ingress, 32).await;
    let keys = seed(addr, 32, VALUE).await;

    let over_before = counter_value(Counter::IngressReplyOverBudget);
    let narrow = Arc::new(mget_command(&keys[..8]));
    let wide = Arc::new(mget_command(&keys));
    let mut tasks = Vec::new();
    for i in 0..NARROW + WIDE {
        let cmd = if i < NARROW {
            Arc::clone(&narrow)
        } else {
            Arc::clone(&wide)
        };
        tasks.push(tokio::spawn(async move {
            let mut s = TcpStream::connect(addr).await.expect("connect");
            s.write_all(&cmd).await.expect("mget");
            read_reply(&mut s, Duration::from_secs(60)).await
        }));
    }

    let mut answered = 0usize;
    let mut refused = 0usize;
    for t in tasks {
        let reply = t.await.expect("task");
        match reply.first() {
            Some(b'*') => answered += 1,
            Some(b'-') => {
                let text = String::from_utf8_lossy(&reply).into_owned();
                assert!(
                    text.starts_with("-ERR ") || text.starts_with("-BACKPRESSURE "),
                    "every refusal must carry the admission classification: {text:?}"
                );
                refused += 1;
            }
            other => panic!("unexpected reply lead byte {other:?}"),
        }
    }

    assert_eq!(
        answered + refused,
        NARROW + WIDE,
        "every connection must get an answer of one kind or the other"
    );
    assert_eq!(
        answered, NARROW,
        "every request inside the allowance must be answered: {answered} of \
         {NARROW}"
    );
    assert_eq!(
        refused, WIDE,
        "every request past the allowance must be REFUSED before its reply is \
         built, not built and then counted as an overshoot: {refused} of \
         {WIDE}"
    );
    assert_eq!(
        counter_value(Counter::IngressReplyOverBudget) - over_before,
        0,
        "a read whose size was reserved before the fetch is never charged \
         after it: {answered} answered, {refused} refused, and the class was \
         still knowingly overshot"
    );
}

// ------------------------------------------------------------ native reads

/// One native response frame.
async fn native_reply(s: &mut TcpStream, within: Duration) -> skeg_proto::Frame {
    let mut parser = FrameParser::new();
    let mut buf = BytesMut::with_capacity(64 * 1024);
    let deadline = Instant::now() + within;
    loop {
        if let Some(frame) = parser.feed(&mut buf).expect("a native frame parses") {
            return frame;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        assert!(!left.is_zero(), "no native reply within {within:?}");
        let n = tokio::time::timeout(left, s.read_buf(&mut buf))
            .await
            .expect("a reply must arrive")
            .expect("read");
        assert!(n > 0, "the server closed with no reply");
    }
}

/// Seed the store over the native wire, so the native reads under test run
/// against a store the native wire itself filled.
async fn native_seed(addr: std::net::SocketAddr, n: usize, value_len: usize) -> Vec<Vec<u8>> {
    let mut s = TcpStream::connect(addr).await.expect("connect");
    let value = vec![b'v'; value_len];
    let mut keys = Vec::with_capacity(n);
    for i in 0..n {
        let key = format!("k{i:04}").into_bytes();
        s.write_all(&skeg_proto::encode_set(
            i as u64,
            &key,
            &value,
            Flags::empty(),
        ))
        .await
        .expect("set");
        let frame = native_reply(&mut s, Duration::from_secs(30)).await;
        assert_eq!(frame.header.op, Op::Ok, "native seed SET {i} failed");
        keys.push(key);
    }
    keys
}

/// A native `MGET` is the same request on the other wire, and B1 asks for the
/// same order there: the sizes first, the reservation second, the fetch last.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_native_mget_over_the_allowance_is_refused_before_the_fetch() {
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(CAP);
    let (addr, _dir) = native_server(&ingress, 8).await;
    let keys = native_seed(addr, 32, VALUE).await;
    let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();

    let before = counter_value(Counter::KvReadBytesFetched);
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&skeg_proto::encode_mget(1, &refs))
        .await
        .expect("mget");
    let frame = native_reply(&mut s, Duration::from_secs(30)).await;

    assert_eq!(
        frame.header.op,
        Op::Err,
        "a native MGET past the allowance must be refused, not answered"
    );
    let err = decode_err_response(&frame.payload).expect("an Err body decodes");
    assert!(
        matches!(
            err.code,
            Some(ErrCode::Backpressure | ErrCode::InvalidRequest)
        ),
        "the refusal must carry the admission classification, not Internal: {err:?}"
    );
    assert_eq!(
        counter_value(Counter::KvReadBytesFetched) - before,
        0,
        "the refusal must precede the fetch"
    );
}

/// A native `GET` goes through the SAME preflighted read as `MGET`: its
/// value's length is taken from the index, reserved, and only then fetched.
///
/// Proven by the counter rather than by a refusal, because on this wire a
/// value large enough to be unanswerable is also too large to write - the
/// frame ceiling a connection's allowance imposes on the way in is the same
/// size as the reservation it imposes on the way out, and native has no
/// `APPEND` to build one incrementally. What the counter shows is that the
/// bytes handed back were fetched through the bounded path and counted there,
/// which is the thing the old `shards.get` route could not do.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_native_get_is_served_through_the_bounded_read() {
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(CAP);
    let (addr, _dir) = native_server(&ingress, 8).await;
    let keys = native_seed(addr, 1, VALUE).await;

    let before = counter_value(Counter::KvReadBytesFetched);
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&skeg_proto::encode_get(1, &keys[0]))
        .await
        .expect("get");
    let frame = native_reply(&mut s, Duration::from_secs(30)).await;
    assert_eq!(frame.header.op, Op::Ok, "a native GET inside the allowance");
    let value = skeg_proto::decode_value_response(&frame.payload).expect("a value body");
    assert_eq!(value.len(), VALUE, "the whole value must come back");
    assert_eq!(
        counter_value(Counter::KvReadBytesFetched) - before,
        VALUE as u64,
        "a native GET must be served through the bounded, counted read"
    );
}

/// `SKEG.VGET` on RESP3 reserves `MAX_VGET_VECTOR_BYTES` before dispatch
/// (audit 19 A1). The native `Op::Vget` answers with the same vector and had
/// no bound at all: B1 asks for the native equivalents to match.
///
/// Proven against an index that does NOT exist, so a real shard call would
/// answer "vector not found" and only a pre-dispatch refusal answers
/// something else.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_native_vget_is_reserved_before_the_shard_call() {
    // A 4 MiB class gives a 1 MiB allowance, under the ~2 MiB charge a 1 MiB
    // vector bound carries.
    let ingress = budget(4 * 1024 * 1024);
    let (addr, _dir) = native_server(&ingress, 8).await;

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&skeg_proto::encode_vget(1, "does-not-exist", 7))
        .await
        .expect("vget");
    let frame = native_reply(&mut s, Duration::from_secs(30)).await;
    assert_eq!(frame.header.op, Op::Err, "a native VGET must be refused");
    let err = decode_err_response(&frame.payload).expect("an Err body decodes");
    assert!(
        matches!(
            err.code,
            Some(ErrCode::Backpressure | ErrCode::InvalidRequest)
        ),
        "this refusal must come from admission, not from the shard resolving \
         (and refusing) an index name - which would prove the shard WAS \
         called: {err:?}"
    );
    assert!(
        !err.message.to_lowercase().contains("not found")
            && !err.message.to_lowercase().contains("vindex"),
        "an admission refusal must not be the shard's answer wearing a \
         different code: {err:?}"
    );
}

// ------------------------------------------- the read must survive a writer

/// audit 25, F1 (CWE-703). The stale-length guard fails CLOSED, which is the
/// right direction - but it used to fail closed on a legitimate request. If
/// another client rewrote the key between this read's measurement and its
/// fetch, the read was refused outright, with no retry and with a message
/// (`... send it in smaller pieces`) that a four-byte `GET` cannot act on.
/// The auditor measured 14 refusals in 4000 reads under a writer alternating
/// 1 KiB and 256 KiB, and 8 in 4000 under `DEL`+`SET`, where an absent key
/// measures as `0` and any value written in the window is therefore refused
/// with certainty.
///
/// A budget is allowed to refuse a request that does not fit. It is not
/// allowed to refuse one that does, because somebody else was writing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_value_rewritten_under_a_read_is_still_answered() {
    // Four thousand reads of real values move `KvReadBytesFetched`, which
    // three tests in this file assert a zero delta on: F3's rule, applied to
    // the test F1 adds.
    let _turn = COUNTER_TESTS.lock().await;
    const READS: usize = 4000;
    // Generous: nothing here may be refused for SIZE, so any refusal at all
    // is the race and not the class.
    let ingress = budget(128 * 1024 * 1024);
    let (addr, _dir) = resp3_server(&ingress, 16).await;

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Writer one: the same key, alternating small and large, so the measured
    // size is stale in both directions.
    let flip_stop = Arc::clone(&stop);
    let flipper = tokio::spawn(async move {
        let mut w = TcpStream::connect(addr).await.expect("connect");
        let small = vec![b's'; 1024];
        let large = vec![b'l'; 256 * 1024];
        let mut big = false;
        while !flip_stop.load(std::sync::atomic::Ordering::Relaxed) {
            let v: &[u8] = if big { &large } else { &small };
            big = !big;
            w.write_all(&resp_array(&[b"SET", b"churn", v]))
                .await
                .expect("set");
            let _ = read_reply(&mut w, Duration::from_secs(30)).await;
        }
    });

    // Writer two: delete and recreate, so the read's measurement can find no
    // key at all - a bound of zero, and a certain refusal for whatever the
    // write puts there.
    let cycle_stop = Arc::clone(&stop);
    let cycler = tokio::spawn(async move {
        let mut w = TcpStream::connect(addr).await.expect("connect");
        let value = vec![b'c'; 64 * 1024];
        while !cycle_stop.load(std::sync::atomic::Ordering::Relaxed) {
            for cmd in [
                resp_array(&[b"DEL", b"cycle"]),
                resp_array(&[b"SET", b"cycle", &value]),
            ] {
                w.write_all(&cmd).await.expect("write");
                let _ = read_reply(&mut w, Duration::from_secs(30)).await;
            }
        }
    });

    let remeasured_before = counter_value(Counter::KvReadRemeasured);
    let mut reader = TcpStream::connect(addr).await.expect("connect");
    let mut spurious: Vec<String> = Vec::new();
    for i in 0..READS {
        let key: &[u8] = if i % 2 == 0 { b"churn" } else { b"cycle" };
        reader
            .write_all(&resp_array(&[b"GET", key]))
            .await
            .expect("get");
        let reply = read_reply(&mut reader, Duration::from_secs(30)).await;
        if reply.first() == Some(&b'-') {
            let text = String::from_utf8_lossy(&reply).into_owned();
            if spurious.len() < 4 {
                spurious.push(text);
            } else {
                spurious.push(String::new());
            }
        }
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = flipper.await;
    let _ = cycler.await;

    // The race has to have HAPPENED, or this test proves nothing: a run in
    // which no measurement ever went stale would pass with the retry removed.
    assert!(
        counter_value(Counter::KvReadRemeasured) > remeasured_before,
        "no read ever found its measurement stale, so the retry path was \
         never exercised and zero spurious errors means nothing"
    );

    let n = spurious.len();
    assert_eq!(
        n,
        0,
        "a read that fits the class must not be refused because another \
         client was writing: {n} of {READS} were, first few: {:?}",
        &spurious[..n.min(4)]
    );
}

// ------------------------------------------- what the preflight itself costs

/// audit 25, F2 (CWE-400), first half. `MGET` had no arity cap at all: the
/// wire cost of a key is about nine bytes (`$1\r\nk\r\n`) and the preflight
/// then builds a `ScopedKey`, a `Bytes`, a per-shard bucket entry and a
/// `u32` for each of them - about twenty times the request, before anything
/// is charged. The cap is the same shape `SKEG.VMSET` already has, refused
/// with the same typed classification.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_mget_past_the_key_cap_is_refused_before_anything_is_allocated() {
    let _turn = COUNTER_TESTS.lock().await;
    let ingress = budget(128 * 1024 * 1024);
    let (addr, _dir) = resp3_server(&ingress, 8).await;

    // Past MAX_MGET_KEYS, and every key absent, so nothing but the cap can
    // refuse it: without one, this is answered with a null per key.
    let keys: Vec<Vec<u8>> = (0..5000).map(|i| format!("k{i}").into_bytes()).collect();

    let before = counter_value(Counter::KvReadBytesFetched);
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&mget_command(&keys)).await.expect("mget");
    let reply = read_reply(&mut s, Duration::from_secs(30)).await;
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with("-ERR "),
        "an MGET past the key cap must be refused by name: {:?}",
        &text[..text.len().min(200)]
    );
    assert!(
        text.contains("4096") && text.contains("5000"),
        "the refusal must name the cap and what was asked: {text:?}"
    );
    assert_eq!(
        counter_value(Counter::KvReadBytesFetched) - before,
        0,
        "the cap must be checked before anything is read"
    );
}

/// audit 25, F2, second half. The preflight's OWN allocation - one
/// `ScopedKey`, one `Bytes`, one bucket entry and one `u32` per key, then the
/// bounded batch's triples - is proportional to the key count and was taken
/// before `grow_to` had granted anything. A wide `MGET` of keys that do not
/// exist reserves almost nothing for its reply (a null is thirty-two bytes of
/// framing) while costing hundreds of bytes per key in the structures that
/// answer it: the exact request an unbudgeted per-key cost makes free.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wide_mget_of_absent_keys_is_charged_for_its_own_preflight() {
    let _turn = COUNTER_TESTS.lock().await;
    // 4 MiB class => 1 MiB per-connection allowance. 4000 absent keys is
    // ~128 KiB of reply framing (admitted under any class) against ~1 MiB of
    // preflight structures (which the allowance cannot cover).
    let ingress = budget(4 * 1024 * 1024);
    let (addr, _dir) = resp3_server(&ingress, 8).await;
    let keys: Vec<Vec<u8>> = (0..4000).map(|i| format!("k{i}").into_bytes()).collect();

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&mget_command(&keys)).await.expect("mget");
    let reply = read_reply(&mut s, Duration::from_secs(30)).await;
    let text = String::from_utf8_lossy(&reply).into_owned();
    assert!(
        text.starts_with('-'),
        "a wide MGET whose preflight cannot be granted must be refused, not \
         served for free: {:?}",
        &text[..text.len().min(200)]
    );
}
