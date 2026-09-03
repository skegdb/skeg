//! P0.5: `/metrics` and `SKEG.STATS` report the same gauges.
//!
//! They did not. `SKEG.STATS` assembled the memory and ingress gauges by hand
//! in its own handler, so an operator scraping Prometheus could not see the
//! budget at all: the two numbers that say whether the server is about to
//! start refusing were visible only to whoever typed a Redis command. And a
//! three-state gauge was emitted only for the state that was true, so an
//! alert on "the budget went unreadable" had to be written with `absent()`,
//! and a dashboard kept showing the previous state for a scrape interval
//! after a transition.
//!
//! The fix is not to copy the assembly into the exporter. It is to delete the
//! assembly: the governor and the budget report their own gauges when asked,
//! and both surfaces ask. Parity by construction rather than by discipline -
//! which is what this file checks.

#![cfg(feature = "metrics-http")]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use skeg_server::Server;
use skeg_server::ingress::IngressBudget;
use skeg_server::memory::{Headroom, MemoryGovernor, MemorySource};

#[derive(Debug)]
struct Fixed(Headroom);

impl MemorySource for Fixed {
    fn headroom(&self) -> Headroom {
        self.0
    }
}

fn ingress_over(h: Headroom) -> Arc<IngressBudget> {
    let governor =
        Arc::new(MemoryGovernor::new(Arc::new(Fixed(h)), None, Some(0)).expect("a governor"));
    Arc::new(IngressBudget::new(
        governor,
        None,
        Some(8 * 1024 * 1024),
        Some(Duration::from_millis(50)),
        u64::from(u32::MAX),
    ))
}

/// Every `name{labels} value` line of a Prometheus text body, keyed by
/// `name{labels}`. Comment lines are dropped.
fn series(body: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.rsplit_once(' ') {
            out.insert(key.to_owned(), value.to_owned());
        }
    }
    out
}

async fn skeg_stats_body(addr: std::net::SocketAddr) -> String {
    let mut conn = TcpStream::connect(addr).await.expect("connect");
    conn.write_all(b"*1\r\n$10\r\nSKEG.STATS\r\n")
        .await
        .expect("write");
    // A bulk reply: `$<len>\r\n<len bytes>\r\n`.
    let mut buf = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let header_end = buf.windows(2).position(|w| w == b"\r\n");
        if let Some(end) = header_end {
            assert_eq!(
                buf.first(),
                Some(&b'$'),
                "SKEG.STATS answered {:?} instead of a bulk",
                String::from_utf8_lossy(&buf[..end])
            );
            let len: usize = String::from_utf8_lossy(&buf[1..end])
                .parse()
                .expect("a bulk length");
            if buf.len() >= end + 2 + len {
                return String::from_utf8_lossy(&buf[end + 2..end + 2 + len]).into_owned();
            }
        }
        let mut chunk = [0u8; 65536];
        match tokio::time::timeout_at(deadline, conn.read(&mut chunk)).await {
            Ok(Ok(0)) | Err(_) | Ok(Err(_)) => panic!("SKEG.STATS did not answer"),
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
        }
    }
}

async fn metrics_body(port: u16) -> String {
    let mut conn = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect to the exporter");
    conn.write_all(b"GET /metrics HTTP/1.0\r\n\r\n")
        .await
        .expect("write");
    let mut body = String::new();
    conn.read_to_string(&mut body).await.expect("read");
    body.split_once("\r\n\r\n")
        .map_or(body.clone(), |(_, b)| b.to_owned())
}

fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    l.local_addr().expect("addr").port()
}

/// One test, one server, on purpose.
///
/// The gauge registry is process-wide and keyed, which is exactly right for a
/// server - one process, one governor, one class - and exactly wrong for two
/// tests each building a server of their own: the second registration
/// replaces the first, and when the second server is dropped its source is
/// pruned and the first test scrapes a dump with no budget in it at all. So
/// the parity checks share one server and run in order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stats_reports_every_gauge_the_metrics_dump_reports() {
    let ingress = ingress_over(Headroom::Known(150 * 1024 * 1024));
    let dir = tempfile::tempdir().expect("tempdir");
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), 1, 0)
        .await
        .expect("bind")
        .with_ingress_budget(Arc::clone(&ingress));
    let addr = server.local_addr().expect("addr");
    tokio::spawn(async move {
        let _ = server.run_resp3().await;
    });

    let metrics_port = free_port();
    std::thread::spawn(move || {
        let _ = skeg_telemetry::http::serve_blocking(
            format!("127.0.0.1:{metrics_port}").parse().expect("addr"),
        );
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    let stats = series(&skeg_stats_body(addr).await);
    let metrics = series(&metrics_body(metrics_port).await);

    assert!(!metrics.is_empty(), "the exporter answered nothing");
    let missing: Vec<&String> = metrics.keys().filter(|k| !stats.contains_key(*k)).collect();
    assert!(
        missing.is_empty(),
        "SKEG.STATS is missing series the /metrics dump reports: {missing:?}"
    );
    for name in [
        "skeg_memory_budget_state{state=\"known\"}",
        "skeg_memory_reserved_bytes",
        "skeg_ingress_state{state=\"known\"}",
        "skeg_ingress_cap_bytes",
        "skeg_ingress_held_bytes",
        "skeg_ingress_per_connection_max_bytes",
    ] {
        assert!(
            metrics.contains_key(name),
            "{name} is not in /metrics, which is where an operator scrapes it"
        );
        assert!(stats.contains_key(name), "{name} is not in SKEG.STATS");
    }

    // The state sets, on the same scrape: all three series present, exactly
    // one of them at 1.
    for (metric, states) in [
        (
            "skeg_memory_budget_state",
            ["known", "unlimited", "unknown"],
        ),
        ("skeg_ingress_state", ["known", "default", "unreadable"]),
    ] {
        let values: Vec<&str> = states
            .iter()
            .map(|s| {
                stats
                    .get(&format!("{metric}{{state=\"{s}\"}}"))
                    .unwrap_or_else(|| {
                        panic!(
                            "{metric} state {s} is absent; an alert on a state \
                             that is only ever emitted when true has to be \
                             written with absent(), and a dashboard shows the \
                             previous state until the series goes stale"
                        )
                    })
                    .as_str()
            })
            .collect();
        assert_eq!(
            values.iter().filter(|v| **v == "1").count(),
            1,
            "{metric}: exactly one state is true, got {values:?}"
        );
    }
}
