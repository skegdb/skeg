//! The one public RESP3 binary serves both profiles, and the bind guard
//! tells them apart.
//!
//! There used to be a second binary for the authenticated profile, in its own
//! crate, which an operator could start by mistake - or, worse, forget to
//! start, serving the unauthenticated engine on the port the multi-tenant
//! deployment was meant to be on. Now `skeg-resp3` is both: no tenant flags
//! is the single profile, `--tenant-auth --tenant-strict` is the
//! authenticated multi-tenant one, and only the second counts as
//! authenticated for the non-loopback guard.

use std::io::Read;
use std::process::{Command, Stdio};

use skeg_tenant::AuthStore;
use skeg_tenant::TenantId;
use skeg_tenant::auth::{Argon2Params, hash_password_with};

/// Run `skeg-server --addr 0.0.0.0:0` (plus `extra_args`) against a scratch
/// data dir, wait for it to exit, and return (success, stderr).
fn run_and_wait(extra_args: &[&str]) -> (bool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skeg-resp3"));
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
fn resp3_binary_without_tenant_auth_refuses_unauthenticated_network_bind() {
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
fn resp3_binary_with_lenient_tenant_auth_refuses_unauthenticated_network_bind() {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth_path = write_auth_store(dir.path());
    let (success, stderr) = run_and_wait(&["--tenant-auth", auth_path.to_str().unwrap()]);
    assert!(!success, "expected non-zero exit; stderr: {stderr}");
    assert!(
        stderr.contains("--allow-unauthenticated-network"),
        "stderr: {stderr}"
    );
}

/// A free port, taken and released: the child binds it a moment later. A
/// port the kernel just handed out is not handed out again while this test
/// runs, and the readiness check below is a TCP connect, not a line of log.
fn free_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral");
    l.local_addr().expect("addr").port()
}

/// Start the binary and wait until it ANSWERS, not until it says it will.
///
/// The old shape of this test scraped "listening on" out of the child's log:
/// a gate that depends on a log line's wording, on the log level, and on the
/// pipe being drained fast enough. A connect loop asks the thing the test
/// actually needs.
fn spawn_ready(args: &[&str], port: u16) -> std::process::Child {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_skeg-resp3"));
    cmd.args(args)
        .env_remove(skeg_server::ALLOW_ENV)
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().expect("spawn binary");
    let addr = format!("127.0.0.1:{port}");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            panic!("server exited before it listened: {status:?}");
        }
        if std::net::TcpStream::connect(&addr).is_ok() {
            return child;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            panic!("server did not accept a connection on {addr} within 20s");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// One RESP3 connection with a reader that survives between commands.
///
/// A `BufReader` built per command would swallow whatever it read past the
/// first line and drop it, so the next reply would never arrive.
struct Conn {
    sock: std::net::TcpStream,
    reader: std::io::BufReader<std::net::TcpStream>,
}

impl Conn {
    fn connect(addr: &str) -> Self {
        let sock = std::net::TcpStream::connect(addr).expect("connect");
        sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .expect("read timeout");
        let reader = std::io::BufReader::new(sock.try_clone().expect("clone"));
        Self { sock, reader }
    }

    /// Send one command as an array of bulk strings; return the first reply
    /// line, trimmed.
    fn cmd(&mut self, args: &[&str]) -> String {
        use std::io::Write;
        let mut out = format!("*{}\r\n", args.len());
        for a in args {
            out.push_str(&format!("${}\r\n{a}\r\n", a.len()));
        }
        self.sock.write_all(out.as_bytes()).expect("write");
        self.sock.flush().expect("flush");
        self.line()
    }

    /// One command on a fresh connection, for a single-shot check.
    fn cmd_once(mut self, args: &[&str]) -> String {
        self.cmd(args)
    }

    /// A RESP3 `HELLO` answers a map of `n` pairs; read the rest of it.
    fn drain_hello(&mut self, header: &str) {
        if let Some(n) = header
            .strip_prefix('%')
            .and_then(|n| n.parse::<usize>().ok())
        {
            for _ in 0..n * 2 {
                let l = self.line();
                if let Some(len) = l.strip_prefix('$').and_then(|n| n.parse::<usize>().ok()) {
                    let _ = len;
                    let _ = self.line();
                }
            }
        }
    }

    fn line(&mut self) -> String {
        use std::io::BufRead;
        let mut line = String::new();
        self.reader.read_line(&mut line).expect("read line");
        line.trim_end().to_owned()
    }
}

#[test]
fn resp3_binary_with_strict_tenant_auth_does_not_check() {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth_path = write_auth_store(dir.path());
    let data_dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();
    // 0.0.0.0 with strict auth: the guard must NOT refuse it, and the proof
    // is that the server accepts a connection.
    let mut child = spawn_ready(
        &[
            "--addr",
            &format!("0.0.0.0:{port}"),
            "--data-dir",
            data_dir.path().to_str().unwrap(),
            "--tenant-auth",
            auth_path.to_str().unwrap(),
            "--tenant-strict",
        ],
        port,
    );
    let _ = child.kill();
    let _ = child.wait();
}

/// The multi-tenant profile, served by the same binary that serves the single
/// one: an anonymous client is refused, two authenticated tenants use the same
/// key name and do not see each other's value.
#[test]
fn one_binary_serves_two_isolated_tenants_and_refuses_the_anonymous() {
    let dir = tempfile::tempdir().expect("tempdir");
    let auth_path = dir.path().join("auth.kdb");
    let mut store = AuthStore::open(&auth_path).expect("open auth store");
    for (user, tenant) in [("alice", "acme"), ("bob", "globex")] {
        let hash = hash_password_with(user.as_bytes(), Argon2Params::default()).expect("hash");
        store.upsert(user, TenantId::from_name(tenant), hash);
    }
    store.save().expect("save auth store");

    let data_dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();
    let mut child = spawn_ready(
        &[
            "--addr",
            &format!("127.0.0.1:{port}"),
            "--data-dir",
            data_dir.path().to_str().unwrap(),
            "--tenant-auth",
            auth_path.to_str().unwrap(),
            "--tenant-strict",
        ],
        port,
    );
    let addr = format!("127.0.0.1:{port}");

    let anon = Conn::connect(&addr).cmd_once(&["HELLO", "3"]);
    assert!(
        anon.starts_with("-NOAUTH"),
        "strict mode must refuse an anonymous HELLO: {anon}"
    );

    let mut sessions = Vec::new();
    for user in ["alice", "bob"] {
        let mut c = Conn::connect(&addr);
        let hello = c.cmd(&["HELLO", "3", "AUTH", user, user]);
        assert!(
            !hello.starts_with('-'),
            "{user} could not authenticate: {hello}"
        );
        // HELLO 3 answers a map; drain it so the next reply is this
        // connection's, not the tail of the handshake.
        c.drain_hello(&hello);
        // Every tenant writes the SAME key name with its own value.
        let set = c.cmd(&["SET", "shared", user]);
        assert_eq!(set, "+OK", "{user} could not write: {set}");
        sessions.push((user, c));
    }
    for (user, c) in &mut sessions {
        let header = c.cmd(&["GET", "shared"]);
        assert_eq!(
            header,
            format!("${}", user.len()),
            "{user} read a value of the wrong length - tenants are not isolated"
        );
        assert_eq!(c.line(), *user, "{user} read another tenant's value");
    }

    let _ = child.kill();
    let _ = child.wait();
}

/// The product rule, pinned where it can be broken: one public RESP3
/// executable, and nothing that tells an operator to start another one.
///
/// A second binary is not a documentation problem - it is a deployment whose
/// authentication depends on which of two commands someone typed. This test
/// reads the manifests, the Dockerfile and the release workflows, so adding
/// one back fails here rather than in production.
#[test]
fn nothing_ships_a_second_resp3_binary() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let shim = std::fs::read_to_string(root.join("crates/skeg-server-tenant/Cargo.toml"))
        .expect("read the compatibility crate's manifest");
    assert!(
        !shim.contains("[[bin]]"),
        "skeg-server-tenant declares a binary again: the multi-tenant profile \
         belongs to skeg-resp3"
    );

    for (file, what) in [
        ("Dockerfile", "the image"),
        (
            ".github/workflows/build-artifacts.yml",
            "the artifact build",
        ),
        (".github/workflows/release.yml", "the release build"),
    ] {
        let text = std::fs::read_to_string(root.join(file)).expect("read");
        // The crate name may still appear in the publish list - the shim is
        // still published. What must not appear is a second server BINARY.
        assert!(
            !text.contains("--bin skeg-server"),
            "{what} builds a second server binary: {file}"
        );
        assert!(
            !text.contains("-p skeg-server-tenant"),
            "{what} builds the compatibility crate as a binary: {file}"
        );
    }

    let dockerfile = std::fs::read_to_string(root.join("Dockerfile")).expect("read Dockerfile");
    assert!(
        dockerfile.contains("skeg-resp3"),
        "the image must carry the one RESP3 binary"
    );
}
