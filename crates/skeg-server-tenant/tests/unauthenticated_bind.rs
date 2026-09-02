//! The tenant binary (`skeg-server`, from crate `skeg-server-tenant`) wraps
//! the same unauthenticated engine as the single-tenant `skeg`/`skeg-resp3`
//! binaries whenever it is run without `--tenant-auth`. It must refuse a
//! non-loopback `--addr` in that mode, exactly like the single-tenant
//! binaries (see `crates/skeg-server/tests/unauthenticated_bind.rs`).
//! `--tenant-auth` alone is not enough: in lenient mode an anonymous
//! `HELLO 3` still maps to tenant ZERO, so the check applies until
//! `--tenant-strict` actually rejects anonymous clients.

use std::io::Read;
use std::process::{Command, Stdio};

use skeg_tenant::AuthStore;
use skeg_tenant::TenantId;
use skeg_tenant::auth::{Argon2Params, hash_password_with};

/// Run `skeg-server --addr 0.0.0.0:0` (plus `extra_args`) against a scratch
/// data dir, wait for it to exit, and return (success, stderr).
fn run_and_wait(extra_args: &[&str]) -> (bool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skeg-server"));
    cmd.arg("--addr")
        .arg("0.0.0.0:0")
        .arg("--data-dir")
        .arg(dir.path())
        .args(extra_args)
        .env_remove(skeg_server::ALLOW_ENV)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn binary");
    // A refused bind exits at once. A regression that lets the server start
    // would otherwise block here forever (it listens until killed), so a
    // process still alive at the deadline is reported as "started".
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let started = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break status.success(),
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                break true;
            }
            None => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    };
    let output = child.wait_with_output().expect("wait for output");
    let mut stderr = String::new();
    let _ = std::io::Cursor::new(&output.stderr).read_to_string(&mut stderr);
    (started, stderr)
}

#[test]
fn tenant_binary_without_tenant_auth_refuses_unauthenticated_network_bind() {
    let (success, stderr) = run_and_wait(&[]);
    assert!(!success, "expected non-zero exit; stderr: {stderr}");
    assert!(
        stderr.contains("--allow-unauthenticated-network"),
        "stderr: {stderr}"
    );
}

/// Write a one-user `auth.kdb` into `dir` and return its path.
fn write_auth_store(dir: &std::path::Path) -> std::path::PathBuf {
    let auth_path = dir.join("auth.kdb");
    let mut store = AuthStore::open(&auth_path).expect("open auth store");
    let hash = hash_password_with(b"pw", Argon2Params::default()).expect("hash password");
    store.upsert("u", TenantId::from_name("acme"), hash);
    store.save().expect("save auth store");
    auth_path
}

#[test]
fn tenant_binary_with_lenient_tenant_auth_refuses_unauthenticated_network_bind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth_path = write_auth_store(dir.path());
    let (success, stderr) = run_and_wait(&["--tenant-auth", auth_path.to_str().unwrap()]);
    assert!(!success, "expected non-zero exit; stderr: {stderr}");
    assert!(
        stderr.contains("--allow-unauthenticated-network"),
        "stderr: {stderr}"
    );
}

#[test]
fn tenant_binary_with_strict_tenant_auth_does_not_check() {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth_path = write_auth_store(dir.path());

    let data_dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skeg-server"));
    cmd.arg("--addr")
        .arg("0.0.0.0:0")
        .arg("--data-dir")
        .arg(data_dir.path())
        .arg("--tenant-auth")
        .arg(&auth_path)
        .arg("--tenant-strict")
        .env_remove(skeg_server::ALLOW_ENV)
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn binary");
    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let (tx, rx) = std::sync::mpsc::channel::<String>();
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

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    let mut combined = String::new();
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            let _ = child.kill();
            panic!("process exited early with {status:?}; output so far: {combined}");
        }
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(chunk) => combined.push_str(&chunk),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {}
        }
        if combined.contains("listening on") {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            panic!("timed out waiting for 'listening on'; output so far: {combined}");
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}
