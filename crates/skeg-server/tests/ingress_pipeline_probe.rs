//! A measurement probe, not a gate: how many pipelined `SKEG.VSET`s per second
//! one RESP3 connection sustains.
//!
//! The ingress budget charges a connection for the capacity its read buffer
//! holds, and a cap that is too tight would show up here first - as a burst
//! that has to stall or serialise one frame per read instead of buffering a
//! whole window per syscall. Ignored by default because it is a number, not
//! an assertion; run it with `--ignored --nocapture` before and after a change
//! to the ingress path and put the pair in the CHANGELOG with their date.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

use skeg_server::Server;

const DIM: usize = 128;
const N: usize = 20_000;

fn encode_bulk(out: &mut Vec<u8>, arg: &[u8]) {
    out.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
    out.extend_from_slice(arg);
    out.extend_from_slice(b"\r\n");
}

#[test]
#[ignore = "measurement probe: run with --ignored --nocapture"]
fn pipelined_vset_throughput() {
    let dir = tempfile::tempdir().expect("tempdir");
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let addr = rt.block_on(async {
        let server = Server::bind("127.0.0.1:0", dir.path()).await.expect("bind");
        let addr = server.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = server.run_resp3().await;
        });
        addr
    });

    let mut sock = TcpStream::connect(addr).expect("connect");
    sock.set_nodelay(true).expect("nodelay");

    // HELLO 3, then create the index, one round trip each.
    let mut hello = Vec::new();
    hello.extend_from_slice(b"*2\r\n");
    encode_bulk(&mut hello, b"HELLO");
    encode_bulk(&mut hello, b"3");
    sock.write_all(&hello).expect("hello");
    let mut scratch = [0u8; 4096];
    let _ = sock.read(&mut scratch).expect("hello reply");

    let mut create = Vec::new();
    create.extend_from_slice(b"*4\r\n");
    encode_bulk(&mut create, b"SKEG.VINDEX.CREATE");
    encode_bulk(&mut create, b"probe");
    encode_bulk(&mut create, DIM.to_string().as_bytes());
    encode_bulk(&mut create, b"flat");
    sock.write_all(&create).expect("create");
    let n = sock.read(&mut scratch).expect("create reply");
    let reply = String::from_utf8_lossy(&scratch[..n]).into_owned();
    assert!(reply.starts_with("+OK"), "index create refused: {reply}");

    // One buffer holding every request, so the write side is not the thing
    // being measured.
    let vector: Vec<u8> = (0..DIM)
        .flat_map(|i| (i as f32 / DIM as f32).to_le_bytes())
        .collect();
    let mut requests = Vec::with_capacity(N * (DIM * 4 + 64));
    for id in 0..N {
        requests.extend_from_slice(b"*4\r\n");
        encode_bulk(&mut requests, b"SKEG.VSET");
        encode_bulk(&mut requests, b"probe");
        encode_bulk(&mut requests, id.to_string().as_bytes());
        encode_bulk(&mut requests, &vector);
    }

    let start = Instant::now();
    let writer = {
        let mut w = sock.try_clone().expect("clone");
        std::thread::spawn(move || {
            w.write_all(&requests).expect("write burst");
            w.flush().expect("flush");
        })
    };
    // Every reply is `+OK\r\n`, so counting the frames is counting the '+'.
    let mut seen = 0usize;
    let mut buf = [0u8; 64 * 1024];
    while seen < N {
        let n = sock.read(&mut buf).expect("read replies");
        assert!(n > 0, "server closed after {seen} replies");
        assert!(
            !buf[..n].contains(&b'-'),
            "the server answered with an error after {seen} replies: {}",
            String::from_utf8_lossy(&buf[..n])
        );
        seen += buf[..n].iter().filter(|&&b| b == b'+').count();
    }
    writer.join().expect("writer");
    let elapsed = start.elapsed();
    println!(
        "pipelined VSET: {N} commands in {:.3} s = {:.0} ops/s (dim {DIM}, one connection)",
        elapsed.as_secs_f64(),
        N as f64 / elapsed.as_secs_f64()
    );
}
