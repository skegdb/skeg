//! The tenant binary (`skeg-server`, from crate `skeg-server-tenant`) wraps
//! the same unauthenticated engine as the single-tenant `skeg`/`skeg-resp3`
//! binaries whenever it is run without `--tenant-auth`. It must refuse a
//! non-loopback `--addr` in that mode, exactly like the single-tenant
//! binaries (see `crates/skeg-server/tests/unauthenticated_bind.rs`). With
//! `--tenant-auth` given, auth is present and no check applies.

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
    let child = cmd.spawn().expect("spawn binary");
    let output = child.wait_with_output().expect("wait for output");
    let mut stderr = String::new();
    let _ = std::io::Cursor::new(&output.stderr).read_to_string(&mut stderr);
    (output.status.success(), stderr)
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

#[test]
fn tenant_binary_with_tenant_auth_does_not_check() {
    // Cheap to build: an `auth.kdb` is just an `AuthStore::open` + `save`.
    let dir = tempfile::tempdir().expect("tempdir");
    let auth_path = dir.path().join("auth.kdb");
    let mut store = AuthStore::open(&auth_path).expect("open auth store");
    let hash = hash_password_with(b"pw", Argon2Params::default()).expect("hash password");
    store.upsert("u", TenantId::from_name("acme"), hash);
    store.save().expect("save auth store");

    let data_dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skeg-server"));
    cmd.arg("--addr")
        .arg("0.0.0.0:0")
        .arg("--data-dir")
        .arg(data_dir.path())
        .arg("--tenant-auth")
        .arg(&auth_path)
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
