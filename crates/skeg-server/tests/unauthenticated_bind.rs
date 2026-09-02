//! Integration coverage for the P0.6 fix: the single-tenant binaries
//! (`skeg`, `skeg-resp3`) have no authentication, so a non-loopback
//! `--addr` must be refused unless `--allow-unauthenticated-network` is
//! passed. This spawns the real binaries end to end.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Run `bin_exe --addr 0.0.0.0:0` with the given extra args against a
/// scratch data dir, wait for it to exit, and return (success, stderr).
fn run_and_wait(bin_exe: &str, extra_args: &[&str]) -> (bool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(bin_exe);
    cmd.arg("--addr")
        .arg("0.0.0.0:0")
        .arg("--data-dir")
        .arg(dir.path())
        .args(extra_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn binary");
    let status = child.wait().expect("wait on child");
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("stderr piped")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    (status.success(), stderr)
}

/// Spawn `bin_exe --addr 0.0.0.0:0 --allow-unauthenticated-network`
/// against a scratch data dir and poll its output for the "listening on"
/// line within a bounded timeout. Always kills the child before
/// returning.
fn assert_starts_with_flag(bin_exe: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(bin_exe);
    cmd.arg("--addr")
        .arg("0.0.0.0:0")
        .arg("--data-dir")
        .arg(dir.path())
        .arg("--allow-unauthenticated-network")
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn binary");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");
    let mut guard = KillOnDrop(child);

    // tracing::info! goes to stdout via the fmt layer; watch both streams
    // in case buffering surfaces it on stderr instead. Each reader runs on
    // its own thread and streams lines back over a channel so the main
    // thread can poll with an overall deadline instead of blocking forever
    // on a `read` that may never return data.
    let (tx, rx) = mpsc::channel::<String>();
    for mut reader in [Box::new(stdout) as Box<dyn Read + Send>, Box::new(stderr)] {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx
                            .send(String::from_utf8_lossy(&buf[..n]).into_owned())
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
        });
    }
    drop(tx);

    let deadline = Instant::now() + Duration::from_secs(15);
    let mut combined = String::new();
    loop {
        if let Some(status) = guard.0.try_wait().expect("try_wait") {
            panic!("process exited early with {status:?}; output so far: {combined}");
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => combined.push_str(&chunk),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {}
        }
        if combined.contains("listening on") {
            break;
        }
        if Instant::now() >= deadline {
            panic!("timed out waiting for 'listening on'; output so far: {combined}");
        }
    }
}

#[test]
fn skeg_refuses_unauthenticated_network_bind_by_default() {
    let (success, stderr) = run_and_wait(env!("CARGO_BIN_EXE_skeg"), &[]);
    assert!(!success, "expected non-zero exit; stderr: {stderr}");
    assert!(
        stderr.contains("--allow-unauthenticated-network"),
        "stderr: {stderr}"
    );
}

#[test]
fn skeg_resp3_refuses_unauthenticated_network_bind_by_default() {
    let (success, stderr) = run_and_wait(env!("CARGO_BIN_EXE_skeg-resp3"), &[]);
    assert!(!success, "expected non-zero exit; stderr: {stderr}");
    assert!(
        stderr.contains("--allow-unauthenticated-network"),
        "stderr: {stderr}"
    );
}

#[test]
fn skeg_starts_with_allow_flag() {
    assert_starts_with_flag(env!("CARGO_BIN_EXE_skeg"));
}

#[test]
fn skeg_resp3_starts_with_allow_flag() {
    assert_starts_with_flag(env!("CARGO_BIN_EXE_skeg-resp3"));
}
