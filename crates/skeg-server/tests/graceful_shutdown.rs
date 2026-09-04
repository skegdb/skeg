use std::io;
use std::time::Duration;

use skeg_core::Durability;
use skeg_server::Server;
use tokio::net::TcpStream;

#[cfg(unix)]
async fn wait_listening(addr: std::net::SocketAddr) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if TcpStream::connect(addr).await.is_ok() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("server listen deadline");
}

#[cfg(unix)]
fn sigterm(child: &std::process::Child) {
    // SAFETY: `id` is the live child we just spawned; kill does not borrow or
    // dereference process memory.
    let rc = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    assert_eq!(rc, 0, "SIGTERM failed: {}", io::Error::last_os_error());
}

#[cfg(unix)]
fn process_server(dir: &std::path::Path, addr: std::net::SocketAddr) -> std::process::Command {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_skeg-resp3"));
    command
        .arg("--addr")
        .arg(addr.to_string())
        .arg("--data-dir")
        .arg(dir)
        .env("SKEG_SHARDS", "1")
        .env("RUST_LOG", "off")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    command
}

#[tokio::test]
async fn clean_shutdown_flushes_joins_and_releases_the_store_lock() {
    let started_before = skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownStarted);
    let completed_before =
        skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownCompleted);
    let dir = tempfile::TempDir::new().unwrap();
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), 2, 0)
        .await
        .unwrap();
    server
        .shards()
        .set(b"confirmed", b"value", Durability::Relaxed)
        .await
        .unwrap();
    server
        .shards()
        .vindex_create("durable", 4, 0, 1)
        .await
        .unwrap();
    server
        .shards()
        .vset("durable", 7, vec![1.0; 4], 0, None, None)
        .await
        .unwrap();

    server
        .run_resp3_until_with_timeout(async { Ok(()) }, Duration::from_secs(2))
        .await
        .expect("clean shutdown barrier");
    assert!(
        skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownStarted) > started_before
    );
    assert!(
        skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownCompleted)
            > completed_before
    );

    let reopened = skeg_server::shard::ShardSet::open(dir.path(), 2).unwrap();
    assert_eq!(
        reopened.get(b"confirmed").await.unwrap().as_deref(),
        Some(b"value".as_slice())
    );
    assert_eq!(
        reopened.vget("durable", 7).await.unwrap(),
        Some(vec![1.0; 4]),
        "the vector delta WAL is part of the shutdown barrier"
    );
}

#[tokio::test]
async fn an_idle_connection_past_the_deadline_makes_shutdown_fail() {
    let failures_before = skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownFailures);
    let timeouts_before =
        skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownConnectionTimeouts);
    let dir = tempfile::TempDir::new().unwrap();
    let server = Server::bind_with_shards("127.0.0.1:0", dir.path(), 1, 0)
        .await
        .unwrap();
    let addr = server.local_addr().unwrap();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
    let run = tokio::spawn(server.run_resp3_until_with_timeout(
        async move {
            stop_rx.await.map_err(io::Error::other)?;
            Ok(())
        },
        Duration::from_millis(50),
    ));
    let _idle = TcpStream::connect(addr).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    stop_tx.send(()).unwrap();

    let err = run.await.unwrap().unwrap_err();
    assert!(
        err.to_string().contains("deadline"),
        "deadline failure was not named: {err}"
    );
    assert!(
        skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownFailures) > failures_before
    );
    assert!(
        skeg_telemetry::counter_value(skeg_telemetry::Counter::ShutdownConnectionTimeouts)
            > timeouts_before
    );
    skeg_server::shard::ShardSet::open(dir.path(), 1)
        .expect("workers were joined and the store lock released");
}

#[cfg(unix)]
#[tokio::test]
async fn sigterm_waits_for_the_barrier_and_exits_zero_when_it_lands() {
    let dir = tempfile::TempDir::new().unwrap();
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = probe.local_addr().unwrap();
    drop(probe);
    let mut command = process_server(dir.path(), addr);
    command.env("SKEG_TEST_SHUTDOWN_FLUSH_DELAY_MS", "300");
    let mut child = command.spawn().unwrap();
    wait_listening(addr).await;
    sigterm(&child);
    tokio::time::sleep(Duration::from_millis(75)).await;
    assert!(
        child.try_wait().unwrap().is_none(),
        "SIGTERM bypassed the controlled flush barrier"
    );
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || child.wait_with_output()),
    )
    .await
    .expect("process shutdown deadline")
    .unwrap()
    .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[tokio::test]
async fn eio_and_enospc_in_the_final_flush_reach_the_process_exit_code() {
    for kind in ["EIO", "ENOSPC"] {
        let dir = tempfile::TempDir::new().unwrap();
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);
        let mut command = process_server(dir.path(), addr);
        command
            .env("SKEG_SHARDS", "2")
            .env("SKEG_TEST_SHUTDOWN_FLUSH_ERROR", kind);
        let child = command.spawn().unwrap();
        wait_listening(addr).await;
        sigterm(&child);
        let output = tokio::time::timeout(
            Duration::from_secs(10),
            tokio::task::spawn_blocking(move || child.wait_with_output()),
        )
        .await
        .expect("process shutdown deadline")
        .unwrap()
        .unwrap();
        assert!(
            !output.status.success(),
            "a failed {kind} flush exited zero"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("final flush"),
            "{kind} was not propagated: {stderr}"
        );
        assert!(
            stderr.contains("shard 0") && stderr.contains("shard 1"),
            "shutdown stopped at the first {kind} shard error: {stderr}"
        );
    }
}

#[cfg(unix)]
#[tokio::test]
async fn connection_deadline_reaches_the_process_exit_code_after_shards_join() {
    let dir = tempfile::TempDir::new().unwrap();
    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = probe.local_addr().unwrap();
    drop(probe);
    let mut command = process_server(dir.path(), addr);
    command.env("SKEG_SHUTDOWN_TIMEOUT_MS", "50");
    let child = command.spawn().unwrap();
    wait_listening(addr).await;
    let _idle = TcpStream::connect(addr).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    sigterm(&child);
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::task::spawn_blocking(move || child.wait_with_output()),
    )
    .await
    .expect("process shutdown deadline")
    .unwrap()
    .unwrap();
    assert!(!output.status.success(), "a drain timeout exited zero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("connection drain deadline expired"),
        "deadline did not reach the process result: {stderr}"
    );
    skeg_server::shard::ShardSet::open(dir.path(), 1)
        .expect("the process joined its shards before returning the failure");
}
