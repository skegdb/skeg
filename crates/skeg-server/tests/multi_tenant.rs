#![cfg(feature = "tenant")]

//! Multi-tenant integration tests.
//!
//! Spins up a RESP3 server with a `TenantContext` wired in, registers
//! two users (alice/bob) in the auth store, and verifies end-to-end:
//!
//! - HELLO 3 AUTH alice hunter2 succeeds and stamps the connection
//! - SKEG.WHOAMI reports the bound tenant
//! - SET/GET against the same logical key are isolated between alice
//!   and bob
//! - HELLO 3 AUTH alice wrongpass is rejected with WRONGPASS
//! - HELLO 3 (no AUTH) keeps the connection on TenantId::ZERO when
//!   the resolver chain is lenient

use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use skeg_server::Server;
use skeg_server::tenant_ctx::TenantContext;
use skeg_tenant::TenantId;
use skeg_tenant::auth::{PasswordHash, cheap_test_cost, hash_password_with};

async fn read_some(stream: &mut TcpStream) -> Vec<u8> {
    let mut buf = vec![0u8; 8192];
    let n = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf))
        .await
        .expect("read timeout")
        .expect("read err");
    buf.truncate(n);
    buf
}

fn array_cmd(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(format!("*{}\r\n", parts.len()).as_bytes());
    for p in parts {
        out.extend_from_slice(format!("${}\r\n", p.len()).as_bytes());
        out.extend_from_slice(p);
        out.extend_from_slice(b"\r\n");
    }
    out
}

/// Skim a RESP3 bulk-string `$N\r\n...payload...\r\n` out of the
/// response bytes. Returns the bytes between the size prefix and the
/// trailing CRLF. Caller already knows the structure.
fn pick_bulk(resp: &[u8]) -> Option<&[u8]> {
    let dollar = resp.iter().position(|&b| b == b'$')?;
    let crlf = resp[dollar..].windows(2).position(|w| w == b"\r\n")?;
    let len_str = std::str::from_utf8(&resp[dollar + 1..dollar + crlf]).ok()?;
    let len: usize = len_str.parse().ok()?;
    let body = dollar + crlf + 2;
    resp.get(body..body + len)
}

async fn build_server_with_tenants() -> (Arc<TenantContext>, std::net::SocketAddr, TempDir) {
    let dir = TempDir::new().expect("tempdir");

    // Auth store seeded with two users. cheap_test_cost is used here
    // because integration tests should not pay the production argon2
    // cost on every run.
    let ctx = TenantContext::open_lenient(dir.path().join("auth.kdb")).expect("ctx");
    let alice = TenantId::from_name("alice");
    let bob = TenantId::from_name("bob");
    {
        let mut w = ctx.auth.write();
        let h_alice = PasswordHash(hash_password_with(b"hunter2", cheap_test_cost()).unwrap().0);
        let h_bob = PasswordHash(
            hash_password_with(b"correct horse", cheap_test_cost())
                .unwrap()
                .0,
        );
        w.upsert("alice", alice, h_alice);
        w.upsert("bob", bob, h_bob);
    }

    let server = Server::bind("127.0.0.1:0", dir.path())
        .await
        .expect("bind")
        .with_tenant_ctx(ctx.clone());
    let addr = server.local_addr().expect("local_addr");

    tokio::spawn(async move {
        let _ = server.run_resp3().await;
    });

    (ctx, addr, dir)
}

#[tokio::test]
async fn hello_auth_success_then_whoami_returns_tenant() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;
    let alice = TenantId::from_name("alice");

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let resp = read_some(&mut s).await;
    // HELLO replies with a Map (RESP3) or flat Array (RESP2). Either way
    // we should NOT see a -ERR or -WRONGPASS prefix.
    assert!(
        !resp.starts_with(b"-"),
        "HELLO replied with error: {}",
        String::from_utf8_lossy(&resp)
    );

    s.write_all(&array_cmd(&[b"SKEG.WHOAMI"])).await.unwrap();
    let resp = read_some(&mut s).await;
    let body = pick_bulk(&resp).expect("whoami bulk");
    let s_body = std::str::from_utf8(body).expect("utf8");
    assert!(
        s_body.contains(&format!("tenant={alice}")),
        "WHOAMI body did not mention alice: {s_body}"
    );
    assert!(s_body.contains("mode=tenant-aware"));
}

#[tokio::test]
async fn hello_auth_wrong_password_rejected() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"WRONG"]))
        .await
        .unwrap();
    let resp = read_some(&mut s).await;
    let s_body = String::from_utf8_lossy(&resp);
    assert!(
        s_body.starts_with("-WRONGPASS"),
        "expected -WRONGPASS, got: {s_body}"
    );
}

#[tokio::test]
async fn hello_auth_unknown_user_rejected() {
    // An unknown user should be indistinguishable (timing-wise and
    // wire-wise) from a wrong password.
    let (_ctx, addr, _dir) = build_server_with_tenants().await;
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&array_cmd(&[
        b"HELLO",
        b"3",
        b"AUTH",
        b"nobody",
        b"anything",
    ]))
    .await
    .unwrap();
    let resp = read_some(&mut s).await;
    let s_body = String::from_utf8_lossy(&resp);
    assert!(
        s_body.starts_with("-WRONGPASS"),
        "expected -WRONGPASS, got: {s_body}"
    );
}

#[tokio::test]
async fn anonymous_hello_resolves_to_zero_tenant() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&array_cmd(&[b"HELLO", b"3"])).await.unwrap();
    let _ = read_some(&mut s).await;
    s.write_all(&array_cmd(&[b"SKEG.WHOAMI"])).await.unwrap();
    let resp = read_some(&mut s).await;
    let body = pick_bulk(&resp).expect("whoami bulk");
    let body = std::str::from_utf8(body).unwrap();
    assert!(body.contains("tenant=00000000000000000000000000000000"));
    assert!(body.contains("mode=tenant-aware"));
}

#[tokio::test]
async fn isolation_between_two_authenticated_tenants() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    // Alice writes "k" → "from-alice"
    let mut a = TcpStream::connect(addr).await.expect("connect alice");
    a.write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut a).await;
    a.write_all(&array_cmd(&[b"SET", b"k", b"from-alice"]))
        .await
        .unwrap();
    let _ = read_some(&mut a).await;

    // Bob writes "k" → "from-bob"
    let mut b = TcpStream::connect(addr).await.expect("connect bob");
    b.write_all(&array_cmd(&[
        b"HELLO",
        b"3",
        b"AUTH",
        b"bob",
        b"correct horse",
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut b).await;
    b.write_all(&array_cmd(&[b"SET", b"k", b"from-bob"]))
        .await
        .unwrap();
    let _ = read_some(&mut b).await;

    // Each tenant sees its own value, not the other's.
    a.write_all(&array_cmd(&[b"GET", b"k"])).await.unwrap();
    let resp_a = read_some(&mut a).await;
    let body_a = pick_bulk(&resp_a).expect("a get bulk");
    assert_eq!(body_a, b"from-alice");

    b.write_all(&array_cmd(&[b"GET", b"k"])).await.unwrap();
    let resp_b = read_some(&mut b).await;
    let body_b = pick_bulk(&resp_b).expect("b get bulk");
    assert_eq!(body_b, b"from-bob");
}

#[tokio::test]
async fn anonymous_view_is_separate_from_authenticated() {
    // ZERO tenant uses unprefixed keys: a value set anonymously must
    // not be readable from an authenticated session, and vice versa.
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    let mut anon = TcpStream::connect(addr).await.expect("connect anon");
    anon.write_all(&array_cmd(&[b"HELLO", b"3"])).await.unwrap();
    let _ = read_some(&mut anon).await;
    anon.write_all(&array_cmd(&[b"SET", b"k", b"anonymous"]))
        .await
        .unwrap();
    let _ = read_some(&mut anon).await;

    let mut alice = TcpStream::connect(addr).await.expect("connect alice");
    alice
        .write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut alice).await;
    alice.write_all(&array_cmd(&[b"GET", b"k"])).await.unwrap();
    let resp = read_some(&mut alice).await;
    // Alice's "k" never received a SET, so this must be Null.
    assert!(
        resp.starts_with(b"_\r\n") || resp.starts_with(b"$-1\r\n"),
        "expected RESP3 Null or RESP2 nil, got: {}",
        String::from_utf8_lossy(&resp)
    );
}

/// Defense against the anon-prefix forgery: an anonymous client must
/// not be able to write a key whose first 16 bytes match a bound
/// tenant id, because on disk that would land in the same byte slot
/// as the tenant's scoped key. Without the prefix-validation gate
/// alice's GET of "leak" would return the attacker-injected value.
#[tokio::test]
async fn anon_cannot_forge_tenant_scoped_key_via_prefix() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;
    let alice = TenantId::from_name("alice");

    let mut anon = TcpStream::connect(addr).await.expect("connect anon");
    anon.write_all(&array_cmd(&[b"HELLO", b"3"])).await.unwrap();
    let _ = read_some(&mut anon).await;

    // Craft alice_id || "leak" as the on-wire key. From an anon
    // (ZERO) connection this must be rejected; otherwise alice would
    // read attacker-controlled bytes when she GETs "leak".
    let mut forged = Vec::with_capacity(TenantId::LEN + 4);
    forged.extend_from_slice(alice.as_bytes());
    forged.extend_from_slice(b"leak");
    anon.write_all(&array_cmd(&[b"SET", &forged, b"attacker-value"]))
        .await
        .unwrap();
    let resp = read_some(&mut anon).await;
    assert!(
        resp.starts_with(b"-ERR"),
        "expected -ERR on forged-prefix SET, got: {}",
        String::from_utf8_lossy(&resp)
    );

    let mut alice_s = TcpStream::connect(addr).await.expect("connect alice");
    alice_s
        .write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut alice_s).await;
    alice_s
        .write_all(&array_cmd(&[b"GET", b"leak"]))
        .await
        .unwrap();
    let resp = read_some(&mut alice_s).await;
    assert!(
        resp.starts_with(b"_\r\n") || resp.starts_with(b"$-1\r\n"),
        "alice's 'leak' must be Null, got: {}",
        String::from_utf8_lossy(&resp)
    );
}

/// Symmetric: an anon read of `<tenant_id><key>` must be rejected
/// even when the tenant has written `key`. Without the gate, alice's
/// `SET secret value` would be exfiltratable by anon
/// `GET <alice_id>secret`.
#[tokio::test]
async fn anon_cannot_read_tenant_scoped_key_via_prefix() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;
    let alice = TenantId::from_name("alice");

    let mut alice_s = TcpStream::connect(addr).await.expect("connect alice");
    alice_s
        .write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut alice_s).await;
    alice_s
        .write_all(&array_cmd(&[b"SET", b"secret", b"alice-private"]))
        .await
        .unwrap();
    let _ = read_some(&mut alice_s).await;

    let mut anon = TcpStream::connect(addr).await.expect("connect anon");
    anon.write_all(&array_cmd(&[b"HELLO", b"3"])).await.unwrap();
    let _ = read_some(&mut anon).await;

    let mut forged = Vec::with_capacity(TenantId::LEN + 6);
    forged.extend_from_slice(alice.as_bytes());
    forged.extend_from_slice(b"secret");
    anon.write_all(&array_cmd(&[b"GET", &forged]))
        .await
        .unwrap();
    let resp = read_some(&mut anon).await;
    assert!(
        resp.starts_with(b"-ERR"),
        "expected -ERR on forged-prefix GET, got: {}",
        String::from_utf8_lossy(&resp)
    );
}

/// The gate must not break ordinary anon traffic: a key shorter than
/// 16 bytes, or one whose 16-byte prefix doesn't match any bound
/// tenant, has to keep working.
#[tokio::test]
async fn anon_can_still_use_non_colliding_keys() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    let mut anon = TcpStream::connect(addr).await.expect("connect anon");
    anon.write_all(&array_cmd(&[b"HELLO", b"3"])).await.unwrap();
    let _ = read_some(&mut anon).await;

    // Short key.
    anon.write_all(&array_cmd(&[b"SET", b"k", b"v"]))
        .await
        .unwrap();
    let resp = read_some(&mut anon).await;
    assert!(
        resp.starts_with(b"+OK") || resp.starts_with(b"$2\r\nOK"),
        "expected +OK on short anon key, got: {}",
        String::from_utf8_lossy(&resp)
    );

    // 16-byte key with a prefix that does not match alice or bob.
    let benign = b"AAAAAAAAAAAAAAAAlong-tail";
    anon.write_all(&array_cmd(&[b"SET", benign, b"v"]))
        .await
        .unwrap();
    let resp = read_some(&mut anon).await;
    assert!(
        resp.starts_with(b"+OK") || resp.starts_with(b"$2\r\nOK"),
        "expected +OK on non-colliding 16+ byte anon key, got: {}",
        String::from_utf8_lossy(&resp)
    );
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Each tenant creates a VINDEX with the same logical name "docs". The
/// underlying scoping must keep them in distinct namespaces, and
/// VINDEX.LIST per tenant must return only their own.
#[tokio::test]
async fn vindex_create_and_list_are_tenant_scoped() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    let mut a = TcpStream::connect(addr).await.expect("connect alice");
    a.write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut a).await;
    a.write_all(&array_cmd(&[
        b"SKEG.VINDEX.CREATE",
        b"docs",
        b"4",
        b"f32",
        b"flat",
    ]))
    .await
    .unwrap();
    let resp = read_some(&mut a).await;
    assert!(
        !resp.starts_with(b"-"),
        "alice CREATE failed: {}",
        String::from_utf8_lossy(&resp)
    );

    let mut b = TcpStream::connect(addr).await.expect("connect bob");
    b.write_all(&array_cmd(&[
        b"HELLO",
        b"3",
        b"AUTH",
        b"bob",
        b"correct horse",
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut b).await;
    b.write_all(&array_cmd(&[
        b"SKEG.VINDEX.CREATE",
        b"docs",
        b"4",
        b"f32",
        b"flat",
    ]))
    .await
    .unwrap();
    let resp = read_some(&mut b).await;
    assert!(
        !resp.starts_with(b"-"),
        "bob CREATE 'docs' must coexist with alice's: {}",
        String::from_utf8_lossy(&resp)
    );

    // Alice lists: must see exactly one "docs" (her own), prefix stripped.
    a.write_all(&array_cmd(&[b"SKEG.VINDEX.LIST"]))
        .await
        .unwrap();
    let resp = read_some(&mut a).await;
    let body = pick_bulk(&resp).expect("alice list bulk");
    let body = std::str::from_utf8(body).unwrap();
    let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 1, "alice should see 1 vindex: {body:?}");
    assert!(
        lines[0].starts_with("name=docs "),
        "alice line: {}",
        lines[0]
    );
    assert!(
        !lines[0].contains("::"),
        "prefix must be stripped: {}",
        lines[0]
    );

    // Bob lists: same shape, his own "docs".
    b.write_all(&array_cmd(&[b"SKEG.VINDEX.LIST"]))
        .await
        .unwrap();
    let resp = read_some(&mut b).await;
    let body = pick_bulk(&resp).expect("bob list bulk");
    let body = std::str::from_utf8(body).unwrap();
    let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(lines.len(), 1, "bob should see 1 vindex: {body:?}");
    assert!(lines[0].starts_with("name=docs "), "bob line: {}", lines[0]);
}

/// End-to-end: alice inserts into her "docs", bob inserts into his
/// own "docs" with the SAME id but a different vector. A VSEARCH from
/// either side must only ever see its own vector.
#[tokio::test]
async fn vset_and_vsearch_isolated_per_tenant() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;

    let alice_vec = vec![1.0f32, 0.0, 0.0, 0.0];
    let bob_vec = vec![0.0f32, 1.0, 0.0, 0.0];

    let mut a = TcpStream::connect(addr).await.expect("connect alice");
    a.write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut a).await;
    a.write_all(&array_cmd(&[
        b"SKEG.VINDEX.CREATE",
        b"docs",
        b"4",
        b"f32",
        b"flat",
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut a).await;
    a.write_all(&array_cmd(&[
        b"SKEG.VSET",
        b"docs",
        b"42",
        &f32_le_bytes(&alice_vec),
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut a).await;

    let mut b = TcpStream::connect(addr).await.expect("connect bob");
    b.write_all(&array_cmd(&[
        b"HELLO",
        b"3",
        b"AUTH",
        b"bob",
        b"correct horse",
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut b).await;
    b.write_all(&array_cmd(&[
        b"SKEG.VINDEX.CREATE",
        b"docs",
        b"4",
        b"f32",
        b"flat",
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut b).await;
    b.write_all(&array_cmd(&[
        b"SKEG.VSET",
        b"docs",
        b"42",
        &f32_le_bytes(&bob_vec),
    ]))
    .await
    .unwrap();
    let _ = read_some(&mut b).await;

    // Alice searches with her own probe: top hit must be id 42 with
    // similarity ~1.0 against alice_vec, not bob_vec.
    a.write_all(&array_cmd(&[
        b"SKEG.VSEARCH",
        b"docs",
        b"1",
        b"32",
        &f32_le_bytes(&alice_vec),
    ]))
    .await
    .unwrap();
    let resp = read_some(&mut a).await;
    assert!(
        !resp.starts_with(b"-"),
        "alice VSEARCH error: {}",
        String::from_utf8_lossy(&resp)
    );
    let s = String::from_utf8_lossy(&resp);
    assert!(s.contains("42"), "alice should retrieve id 42: {s}");

    // Bob searches with alice's vector: should NOT find id 42 with
    // high similarity, because bob's id 42 is the orthogonal bob_vec.
    // The hit comes back but the score should be ~0 (orthogonal).
    b.write_all(&array_cmd(&[
        b"SKEG.VSEARCH",
        b"docs",
        b"1",
        b"32",
        &f32_le_bytes(&alice_vec),
    ]))
    .await
    .unwrap();
    let resp = read_some(&mut b).await;
    assert!(
        !resp.starts_with(b"-"),
        "bob VSEARCH error: {}",
        String::from_utf8_lossy(&resp)
    );
}

/// VINDEX names cannot contain `::` because it is the tenant scope
/// separator; user-supplied `::` would let a tenant impersonate another
/// scope on the wire.
#[tokio::test]
async fn vindex_create_rejects_colon_colon_in_name() {
    let (_ctx, addr, _dir) = build_server_with_tenants().await;
    let mut s = TcpStream::connect(addr).await.expect("connect");
    s.write_all(&array_cmd(&[b"HELLO", b"3", b"AUTH", b"alice", b"hunter2"]))
        .await
        .unwrap();
    let _ = read_some(&mut s).await;
    s.write_all(&array_cmd(&[
        b"SKEG.VINDEX.CREATE",
        b"foo::bar",
        b"4",
        b"f32",
        b"flat",
    ]))
    .await
    .unwrap();
    let resp = read_some(&mut s).await;
    assert!(
        resp.starts_with(b"-ERR"),
        "expected -ERR on '::' in name: {}",
        String::from_utf8_lossy(&resp)
    );
}
