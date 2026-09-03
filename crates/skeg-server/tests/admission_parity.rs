//! P0.5: the same refusal, said the same way, on both wires.
//!
//! A refusal taken before the work has exactly one thing a client needs from
//! it: whether sending the same request again is worth doing. The RESP3 wire
//! has carried that since the ingress budget landed - the first word of the
//! error line is `BACKPRESSURE` or `ERR` - and the native wire carried nothing
//! at all: every refusal arrived as `ErrCode::Internal`, which tells a caller
//! to give up, with the word `BACKPRESSURE` buried in prose at byte 16 for
//! the two conditions that bothered to write it.
//!
//! So this file is a TABLE. One row per condition a request can be refused
//! for, driven over a real socket on each wire, asserting that the two wires
//! agree about retryability and that each spells it in its own alphabet. A
//! row that genuinely cannot happen on one wire says so by name and is
//! asserted to be absent there, rather than quietly not being run.
//!
//! Counters are asserted as DELTAS. They are process-wide statics and this
//! binary runs its tests in parallel, so an absolute value is a number some
//! other test is also writing to.

use std::sync::Arc;
use std::time::Duration;

use bytes::BytesMut;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use skeg_proto::{ErrCode, Frame, FrameParser, decode_err_response};
use skeg_server::Server;
use skeg_server::admission::Retryability;
use skeg_server::failpoint::{
    AdmissionFailpoint, IngressFailpoint, arm_admission_at, arm_ingress_at, disarm_admission_at,
    disarm_ingress_at, fired_admission_at, fired_ingress_at,
};
use skeg_server::ingress::{FLOOR_BYTES, IngressBudget, PARSE_FACTOR};
use skeg_server::memory::{Headroom, MemoryGovernor, MemorySource};
use skeg_server::tenant::{Admission, AdmitGuard, AdmitRejected, TenantBackend, TenantId};

// ---------------------------------------------------------------- fixtures

#[derive(Debug)]
struct Fixed(Headroom);

impl MemorySource for Fixed {
    fn headroom(&self) -> Headroom {
        self.0
    }
}

/// A budget whose class cap is exactly `cap`, over headroom eight times that
/// so the governor is never the thing refusing.
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

/// A comfortable budget: nothing here refuses for want of room.
fn roomy() -> Arc<IngressBudget> {
    budget(64 * 1024 * 1024, Duration::from_millis(50))
}

struct Running {
    addr: std::net::SocketAddr,
    _dir: tempfile::TempDir,
}

impl Running {
    /// The failpoint key for this listener: its port. Every server here binds
    /// port 0, so no two share one - the isolation rule the keyed registries
    /// need and that a hand-chosen name cannot guarantee.
    fn key(&self) -> String {
        self.addr.port().to_string()
    }
}

async fn resp3_server(ingress: &Arc<IngressBudget>, max_connections: usize) -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), 1, 0)
        .await
        .expect("bind")
        .with_ingress_budget(Arc::clone(ingress))
        .with_max_connections(max_connections);
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run_resp3().await;
    });
    Running { addr, _dir: dir }
}

async fn resp3_server_with_backend(
    ingress: &Arc<IngressBudget>,
    backend: Arc<dyn TenantBackend>,
) -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), 1, 0)
        .await
        .expect("bind")
        .with_ingress_budget(Arc::clone(ingress))
        .with_tenant_backend(backend)
        .with_max_connections(16);
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run_resp3().await;
    });
    Running { addr, _dir: dir }
}

async fn native_server(ingress: &Arc<IngressBudget>, max_connections: usize) -> Running {
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), 1, 0)
        .await
        .expect("bind")
        .with_ingress_budget(Arc::clone(ingress))
        .with_max_connections(max_connections);
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run().await;
    });
    Running { addr, _dir: dir }
}

// ------------------------------------------------------------ wire helpers

/// Read one RESP3 error line, or `None` if the peer said nothing before EOF.
async fn read_resp3_line(stream: &mut TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let mut byte = [0u8; 1];
        let read = tokio::time::timeout_at(deadline, stream.read(&mut byte)).await;
        match read {
            Ok(Ok(0)) | Err(_) => break,
            Ok(Ok(_)) => {
                buf.push(byte[0]);
                if buf.ends_with(b"\r\n") {
                    break;
                }
            }
            Ok(Err(_)) => break,
        }
    }
    if buf.is_empty() {
        return None;
    }
    Some(String::from_utf8_lossy(&buf).trim_end().to_owned())
}

/// Read one native frame, or `None` if the peer closed without sending one -
/// which is the silence this work exists to replace.
async fn read_native_frame(stream: &mut TcpStream) -> Option<Frame> {
    let mut parser = FrameParser::new();
    let mut buf = BytesMut::with_capacity(4096);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(Some(frame)) = parser.feed(&mut buf) {
            return Some(frame);
        }
        match tokio::time::timeout_at(deadline, stream.read_buf(&mut buf)).await {
            Ok(Ok(0)) | Err(_) | Ok(Err(_)) => return None,
            Ok(Ok(_)) => {}
        }
    }
}

/// One refusal as it arrived, in the terms the wire uses.
#[derive(Debug)]
struct Refusal {
    /// RESP3: the first word of the error line. Native: `None`.
    code_word: Option<String>,
    /// Native: the error code byte, decoded. RESP3: `None`.
    code: Option<Option<ErrCode>>,
    message: String,
}

impl Refusal {
    fn from_resp3_line(line: &str) -> Self {
        let body = line.strip_prefix('-').unwrap_or(line);
        let word = body.split_whitespace().next().unwrap_or("").to_owned();
        Self {
            code_word: Some(word),
            code: None,
            message: body.to_owned(),
        }
    }

    fn from_native_frame(frame: &Frame) -> Self {
        assert_eq!(
            frame.header.op,
            skeg_proto::Op::Err,
            "a refusal must come back as an Err frame"
        );
        let decoded = decode_err_response(&frame.payload).expect("an Err body decodes");
        Self {
            code_word: None,
            code: Some(decoded.code),
            message: decoded.message,
        }
    }

    /// What the wire said about retrying.
    fn retryability(&self) -> Retryability {
        let retryable = match (&self.code_word, &self.code) {
            (Some(word), None) => word == "BACKPRESSURE",
            (None, Some(code)) => code.is_some_and(ErrCode::is_retryable),
            _ => unreachable!("a refusal comes from exactly one wire"),
        };
        if retryable {
            Retryability::Retryable
        } else {
            Retryability::Permanent
        }
    }
}

fn resp3_command(args: &[&[u8]]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", args.len()).into_bytes();
    for a in args {
        out.extend_from_slice(format!("${}\r\n", a.len()).as_bytes());
        out.extend_from_slice(a);
        out.extend_from_slice(b"\r\n");
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// A four-dimensional flat index, asserted created.
async fn create_index(conn: &mut TcpStream, name: &str) {
    let _ = conn
        .write_all(&resp3_command(&[
            b"SKEG.VINDEX.CREATE",
            name.as_bytes(),
            b"4",
            b"flat",
        ]))
        .await;
    let created = read_resp3_line(conn).await.expect("create replies");
    assert!(created.starts_with('+'), "create said {created}");
}

// ---------------------------------------------------------------- the table

/// Every condition a request can be refused for before it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Condition {
    /// The ingress class had no room for a new connection's floor.
    ClassFullAtAccept,
    /// The class had no room for the buffer a frame in flight needed.
    ClassFullMidFrame,
    /// The frame is larger than one connection may ever hold.
    OverConnectionAllowance,
    /// The memory governor refused the heap the write would take.
    MemoryRefusedAtVset,
    /// The tenant is at its vector limit.
    QuotaRefusedAtVset,
    /// One command declared more items than its ceiling allows.
    RequestTooLarge,
    /// The control row: a vector of the wrong dimension. Not admission at
    /// all - it is the request being wrong - and it must stay permanent on
    /// both wires whatever the classification does, or a table where
    /// everything is retryable would look correct.
    DimMismatch,
    /// The bounded VSEARCH pool had no permit left.
    VsearchQueueFull,
    /// A tenant backend refused, with the code word its own contract asks
    /// for.
    BackendRateLimited,
    /// A tenant backend refused with a message the engine cannot classify.
    BackendUnclassified,
}

impl Condition {
    const ALL: [Condition; 10] = [
        Condition::ClassFullAtAccept,
        Condition::ClassFullMidFrame,
        Condition::OverConnectionAllowance,
        Condition::MemoryRefusedAtVset,
        Condition::QuotaRefusedAtVset,
        Condition::RequestTooLarge,
        Condition::DimMismatch,
        Condition::VsearchQueueFull,
        Condition::BackendRateLimited,
        Condition::BackendUnclassified,
    ];

    /// What both wires must say. Exhaustive: a new condition does not compile
    /// until it is classified here.
    fn expected(self) -> (Retryability, ErrCode) {
        match self {
            Condition::ClassFullAtAccept
            | Condition::ClassFullMidFrame
            | Condition::MemoryRefusedAtVset
            | Condition::VsearchQueueFull
            | Condition::BackendRateLimited => (Retryability::Retryable, ErrCode::Backpressure),
            Condition::OverConnectionAllowance
            | Condition::QuotaRefusedAtVset
            | Condition::RequestTooLarge
            | Condition::DimMismatch
            | Condition::BackendUnclassified => (Retryability::Permanent, ErrCode::InvalidRequest),
        }
    }

    /// The word a RESP3 client sees at the front of the line.
    ///
    /// Usually the one the classification chose. A tenant backend writes its
    /// OWN complete error line, code word included, and the engine passes it
    /// through: replacing `RATELIMITED` with `BACKPRESSURE` would take away
    /// the word that deployment's clients already route on. So those rows
    /// pin the word verbatim, and their retryability is checked where it is
    /// observable - the unit tests in `admission.rs`, and the native code
    /// byte if that wire ever carries a backend.
    fn resp3_word(self) -> &'static str {
        match self {
            Condition::ClassFullAtAccept
            | Condition::ClassFullMidFrame
            | Condition::MemoryRefusedAtVset
            | Condition::VsearchQueueFull => "BACKPRESSURE",
            Condition::OverConnectionAllowance
            | Condition::QuotaRefusedAtVset
            | Condition::RequestTooLarge
            | Condition::DimMismatch => "ERR",
            Condition::BackendRateLimited => "RATELIMITED",
            Condition::BackendUnclassified => "TENANTBLOCKED",
        }
    }

    /// The counter this condition must move, if any.
    fn counter(self) -> Option<skeg_telemetry::Counter> {
        match self {
            Condition::ClassFullAtAccept => Some(skeg_telemetry::Counter::IngressRefusedAccept),
            Condition::ClassFullMidFrame => Some(skeg_telemetry::Counter::IngressRefusedGrowth),
            Condition::MemoryRefusedAtVset => Some(skeg_telemetry::Counter::MemoryRefused),
            Condition::QuotaRefusedAtVset => Some(skeg_telemetry::Counter::QuotaRefused),
            Condition::BackendUnclassified => {
                Some(skeg_telemetry::Counter::BackendRefusalUnclassified)
            }
            Condition::OverConnectionAllowance
            | Condition::RequestTooLarge
            | Condition::DimMismatch
            | Condition::VsearchQueueFull
            | Condition::BackendRateLimited => None,
        }
    }
}

/// A tenant backend that refuses every command with a fixed message.
///
/// The engine has no backend of its own - one is supplied by a separate
/// crate - so the only way to exercise the refusal path is to be one.
#[derive(Debug)]
struct RefusingBackend(&'static str);

impl TenantBackend for RefusingBackend {
    fn verify_login(&self, _user: &str, _password: &[u8]) -> Option<TenantId> {
        None
    }

    fn has_tenant(&self, _id: TenantId) -> bool {
        false
    }

    fn admit(&self, _admission: Admission) -> Result<AdmitGuard, AdmitRejected> {
        Err(AdmitRejected {
            message: self.0.to_owned(),
        })
    }
}

/// A condition's outcome on one wire: either a refusal, or a stated reason
/// the condition cannot arise there.
enum Outcome {
    Refused(Refusal),
    NotOnThisWire(&'static str),
}

// ------------------------------------------------------------- RESP3 driver

async fn drive_resp3(cond: Condition) -> Outcome {
    match cond {
        Condition::ClassFullAtAccept => {
            // A class with room for exactly one connection's floor: the
            // second connection is refused before it is served.
            let ingress = budget(FLOOR_BYTES, Duration::from_millis(50));
            let server = resp3_server(&ingress, 16).await;
            let _held = TcpStream::connect(server.addr).await.expect("first");
            let mut refused = TcpStream::connect(server.addr).await.expect("second");
            let line = read_resp3_line(&mut refused)
                .await
                .expect("the peer must be told, not dropped");
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::ClassFullMidFrame => {
            let ingress = budget(64 * 1024, Duration::from_millis(50));
            let server = resp3_server(&ingress, 16).await;
            let key = server.key();
            arm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &key);
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let line = read_resp3_line(&mut conn).await.expect("a refusal");
            disarm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &key);
            assert!(
                fired_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &key),
                "the failpoint never fired, so this row proved nothing"
            );
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::OverConnectionAllowance => {
            // A class whose per-connection quarter is small, and a bulk that
            // declares more than that quarter. Waiting cannot make it fit.
            let ingress = budget(256 * 1024, Duration::from_millis(50));
            let server = resp3_server(&ingress, 16).await;
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let declared = 8 * 1024 * 1024;
            let mut head = format!("*3\r\n$3\r\nSET\r\n$1\r\nk\r\n${declared}\r\n").into_bytes();
            head.extend_from_slice(&vec![b'x'; 64 * 1024]);
            let _ = conn.write_all(&head).await;
            let line = read_resp3_line(&mut conn).await.expect("a refusal");
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::MemoryRefusedAtVset | Condition::QuotaRefusedAtVset => {
            let ingress = roomy();
            let server = resp3_server(&ingress, 16).await;
            let name = format!("parity_resp3_{}", server.addr.port());
            let fp = admission_failpoint(cond);
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            create_index(&mut conn, &name).await;
            arm_admission_at(fp, &name);
            let _ = conn
                .write_all(&resp3_command(&[
                    b"SKEG.VSET",
                    name.as_bytes(),
                    b"1",
                    &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                ]))
                .await;
            let line = read_resp3_line(&mut conn).await.expect("vset replies");
            disarm_admission_at(fp, &name);
            assert!(
                fired_admission_at(fp, &name),
                "the failpoint never fired, so this row proved nothing"
            );
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::RequestTooLarge => {
            let ingress = roomy();
            let server = resp3_server(&ingress, 16).await;
            let name = format!("parity_big_{}", server.addr.port());
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let vector = f32_bytes(&[1.0]);
            let mut args: Vec<Vec<u8>> = vec![b"SKEG.VMSET".to_vec(), name.as_bytes().to_vec()];
            for id in 0..4097u64 {
                args.push(id.to_string().into_bytes());
                args.push(vector.clone());
                args.push(Vec::new());
            }
            let borrowed: Vec<&[u8]> = args.iter().map(Vec::as_slice).collect();
            let _ = conn.write_all(&resp3_command(&borrowed)).await;
            let line = read_resp3_line(&mut conn).await.expect("vmset replies");
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::VsearchQueueFull => {
            let ingress = roomy();
            let server = resp3_server(&ingress, 16).await;
            let name = format!("parity_queue_{}", server.addr.port());
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            create_index(&mut conn, &name).await;
            arm_admission_at(AdmissionFailpoint::VsearchQueueFullAtSearch, &name);
            let _ = conn
                .write_all(&resp3_command(&[
                    b"SKEG.VSEARCH",
                    name.as_bytes(),
                    b"1",
                    b"0",
                    &f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
                ]))
                .await;
            let line = read_resp3_line(&mut conn).await.expect("vsearch replies");
            disarm_admission_at(AdmissionFailpoint::VsearchQueueFullAtSearch, &name);
            assert!(
                fired_admission_at(AdmissionFailpoint::VsearchQueueFullAtSearch, &name),
                "the failpoint never fired, so this row proved nothing"
            );
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::BackendRateLimited | Condition::BackendUnclassified => {
            let message = if cond == Condition::BackendRateLimited {
                "RATELIMITED tenant request rate exceeded"
            } else {
                "TENANTBLOCKED this tenant is suspended"
            };
            let ingress = roomy();
            let server =
                resp3_server_with_backend(&ingress, Arc::new(RefusingBackend(message))).await;
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            // Any command that is not HELLO or SKEG.AUTH goes through the
            // backend's admission, and this one refuses all of them.
            let _ = conn.write_all(&resp3_command(&[b"PING"])).await;
            let line = read_resp3_line(&mut conn).await.expect("ping replies");
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
        Condition::DimMismatch => {
            let ingress = roomy();
            let server = resp3_server(&ingress, 16).await;
            let name = format!("parity_dim_{}", server.addr.port());
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let _ = conn
                .write_all(&resp3_command(&[
                    b"SKEG.VINDEX.CREATE",
                    name.as_bytes(),
                    b"4",
                    b"flat",
                ]))
                .await;
            let created = read_resp3_line(&mut conn).await.expect("create replies");
            assert!(created.starts_with('+'), "create said {created}");
            let _ = conn
                .write_all(&resp3_command(&[
                    b"SKEG.VSET",
                    name.as_bytes(),
                    b"1",
                    &f32_bytes(&[1.0, 2.0, 3.0]),
                ]))
                .await;
            let line = read_resp3_line(&mut conn).await.expect("vset replies");
            Outcome::Refused(Refusal::from_resp3_line(&line))
        }
    }
}

// ------------------------------------------------------------ native driver

fn admission_failpoint(cond: Condition) -> AdmissionFailpoint {
    match cond {
        Condition::MemoryRefusedAtVset => AdmissionFailpoint::MemoryRefusedAtVset,
        Condition::QuotaRefusedAtVset => AdmissionFailpoint::QuotaRefusedAtVset,
        other => unreachable!("{other:?} is not driven by an admission failpoint"),
    }
}

async fn drive_native(cond: Condition) -> Outcome {
    match cond {
        Condition::ClassFullAtAccept => {
            let ingress = budget(FLOOR_BYTES, Duration::from_millis(50));
            let server = native_server(&ingress, 16).await;
            let _held = TcpStream::connect(server.addr).await.expect("first");
            let mut refused = TcpStream::connect(server.addr).await.expect("second");
            let frame = read_native_frame(&mut refused)
                .await
                .expect("the peer must be told, not dropped");
            Outcome::Refused(Refusal::from_native_frame(&frame))
        }
        Condition::ClassFullMidFrame => {
            let ingress = budget(64 * 1024, Duration::from_millis(50));
            let server = native_server(&ingress, 16).await;
            let key = server.key();
            arm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &key);
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let frame = read_native_frame(&mut conn).await.expect("a refusal");
            disarm_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &key);
            assert!(
                fired_ingress_at(IngressFailpoint::GrowRefusedMidFrame, &key),
                "the failpoint never fired, so this row proved nothing"
            );
            Outcome::Refused(Refusal::from_native_frame(&frame))
        }
        Condition::OverConnectionAllowance => {
            // The native form of the same refusal is taken on the HEADER: 24
            // bytes declare a payload larger than the parser's limit, and
            // nothing is buffered. It used to close the socket without a word.
            let ingress = budget(256 * 1024, Duration::from_millis(50));
            let server = native_server(&ingress, 16).await;
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let _ = conn.write_all(&oversized_header(32 * 1024 * 1024)).await;
            let frame = read_native_frame(&mut conn)
                .await
                .expect("refused BY NAME, not by silence");
            Outcome::Refused(Refusal::from_native_frame(&frame))
        }
        Condition::MemoryRefusedAtVset | Condition::QuotaRefusedAtVset => {
            let ingress = roomy();
            let server = native_server(&ingress, 16).await;
            let name = format!("parity_native_{}", server.addr.port());
            let fp = admission_failpoint(cond);
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let _ = conn
                .write_all(&skeg_proto::encode_vindex_create(1, &name, 4, 0, 0))
                .await;
            let created = read_native_frame(&mut conn).await.expect("create replies");
            assert_eq!(created.header.op, skeg_proto::Op::Ok, "create must succeed");
            arm_admission_at(fp, &name);
            let _ = conn
                .write_all(&skeg_proto::encode_vset(
                    2,
                    &name,
                    1,
                    &[1.0, 2.0, 3.0, 4.0],
                    skeg_proto::Flags::empty(),
                ))
                .await;
            let frame = read_native_frame(&mut conn).await.expect("vset replies");
            disarm_admission_at(fp, &name);
            assert!(
                fired_admission_at(fp, &name),
                "the failpoint never fired, so this row proved nothing"
            );
            Outcome::Refused(Refusal::from_native_frame(&frame))
        }
        Condition::RequestTooLarge => Outcome::NotOnThisWire(
            "the native protocol has no VMSET: one frame carries one vector, \
             and its size is bounded by the frame ceiling instead",
        ),
        Condition::BackendRateLimited | Condition::BackendUnclassified => Outcome::NotOnThisWire(
            "the native listener carries no tenant backend - Server::run \
                 drops it and every request is tenant 0 - so a backend \
                 refusal cannot reach this wire at all. The classification is \
                 pinned by the unit tests in admission.rs instead",
        ),
        Condition::VsearchQueueFull => {
            let ingress = roomy();
            let server = native_server(&ingress, 16).await;
            let name = format!("parity_nqueue_{}", server.addr.port());
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let _ = conn
                .write_all(&skeg_proto::encode_vindex_create(1, &name, 4, 0, 0))
                .await;
            let created = read_native_frame(&mut conn).await.expect("create replies");
            assert_eq!(created.header.op, skeg_proto::Op::Ok, "create must succeed");
            arm_admission_at(AdmissionFailpoint::VsearchQueueFullAtSearch, &name);
            let _ = conn
                .write_all(&skeg_proto::encode_vsearch(
                    2,
                    &name,
                    1,
                    &[1.0, 2.0, 3.0, 4.0],
                ))
                .await;
            let frame = read_native_frame(&mut conn).await.expect("vsearch replies");
            disarm_admission_at(AdmissionFailpoint::VsearchQueueFullAtSearch, &name);
            assert!(
                fired_admission_at(AdmissionFailpoint::VsearchQueueFullAtSearch, &name),
                "the failpoint never fired, so this row proved nothing"
            );
            Outcome::Refused(Refusal::from_native_frame(&frame))
        }
        Condition::DimMismatch => {
            let ingress = roomy();
            let server = native_server(&ingress, 16).await;
            let name = format!("parity_ndim_{}", server.addr.port());
            let mut conn = TcpStream::connect(server.addr).await.expect("connect");
            let _ = conn
                .write_all(&skeg_proto::encode_vindex_create(1, &name, 4, 0, 0))
                .await;
            let created = read_native_frame(&mut conn).await.expect("create replies");
            assert_eq!(created.header.op, skeg_proto::Op::Ok, "create must succeed");
            let _ = conn
                .write_all(&skeg_proto::encode_vset(
                    2,
                    &name,
                    1,
                    &[1.0, 2.0, 3.0],
                    skeg_proto::Flags::empty(),
                ))
                .await;
            let frame = read_native_frame(&mut conn).await.expect("vset replies");
            Outcome::Refused(Refusal::from_native_frame(&frame))
        }
    }
}

/// A bare native header declaring `payload_len` bytes and carrying none.
fn oversized_header(payload_len: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(skeg_proto::HEADER_LEN);
    out.extend_from_slice(&skeg_proto::MAGIC.to_le_bytes());
    out.push(skeg_proto::VERSION_V1);
    out.push(0x80); // Ping
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend_from_slice(&7u64.to_le_bytes()); // req_id
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out
}

// ----------------------------------------------------------------- the test

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_admission_refusal_says_the_same_thing_on_both_wires() {
    for cond in Condition::ALL {
        let (want_retryability, want_code) = cond.expected();

        let before = cond.counter().map(skeg_telemetry::counter_value);
        let resp3 = drive_resp3(cond).await;
        let after = cond.counter().map(skeg_telemetry::counter_value);

        match &resp3 {
            Outcome::Refused(r) => {
                // The WORD, not the classification derived from it. Most rows
                // are the same statement twice; the backend rows are not, and
                // that difference is the contract: the engine passes a
                // backend's own line through, so RESP3 shows `RATELIMITED`
                // where the native code would be 0x04.
                assert_eq!(
                    r.code_word.as_deref(),
                    Some(cond.resp3_word()),
                    "{cond:?} on RESP3: {:?}",
                    r.message
                );
                if cond.resp3_word() == "BACKPRESSURE" || cond.resp3_word() == "ERR" {
                    assert_eq!(
                        r.retryability(),
                        want_retryability,
                        "{cond:?} on RESP3: {:?} says {:?}",
                        r.message,
                        r.retryability()
                    );
                }
            }
            Outcome::NotOnThisWire(why) => {
                panic!("{cond:?} was declared absent from RESP3 ({why}) - unexpected")
            }
        }
        if let (Some(b), Some(a)) = (before, after) {
            assert!(
                a > b,
                "{cond:?} on RESP3 moved no counter (delta {}, and it must be \
                 at least one; absolute values are shared with every other \
                 test in this binary)",
                a - b
            );
        }

        let before = cond.counter().map(skeg_telemetry::counter_value);
        let native = drive_native(cond).await;
        let after = cond.counter().map(skeg_telemetry::counter_value);

        match &native {
            Outcome::Refused(r) => {
                assert_eq!(
                    r.retryability(),
                    want_retryability,
                    "{cond:?} on the native wire: {:?} says {:?}",
                    r.message,
                    r.retryability()
                );
                assert_eq!(
                    r.code,
                    Some(Some(want_code)),
                    "{cond:?} on the native wire carried the wrong code for {:?}",
                    r.message
                );
                if let (Some(b), Some(a)) = (before, after) {
                    assert!(a > b, "{cond:?} on the native wire moved no counter");
                }
            }
            Outcome::NotOnThisWire(why) => {
                // Declared, not skipped: the reason is part of the contract.
                assert!(!why.is_empty());
            }
        }
    }
}

/// The one the audit named: a frame the connection may never hold used to
/// close the socket without a word, which a client reads as a network fault
/// and answers with a reconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_frame_over_the_connection_allowance_is_refused_by_name_not_by_silence() {
    let ingress = budget(256 * 1024, Duration::from_millis(50));
    let server = native_server(&ingress, 16).await;
    let mut conn = TcpStream::connect(server.addr).await.expect("connect");
    let allowance = 256 * 1024 / 4 / PARSE_FACTOR;
    let declared = u32::try_from(allowance * 8).expect("fits");
    let _ = conn.write_all(&oversized_header(declared)).await;
    let frame = read_native_frame(&mut conn)
        .await
        .expect("the peer must be told which limit it crossed");
    let refusal = Refusal::from_native_frame(&frame);
    assert_eq!(
        refusal.code,
        Some(Some(ErrCode::InvalidRequest)),
        "a frame this connection may never hold will not fit on a retry either"
    );
    assert!(
        refusal.message.contains("too large") || refusal.message.contains("at most"),
        "the refusal must name the limit; it said {:?}",
        refusal.message
    );
}
