//! Connection pooling for RESP3 transports (F.54).
//!
//! `Resp3Tenant` used to own one connection wrapped in a Mutex.
//! Concurrent calls to the same peer serialised on that lock, and
//! when one hansa membership covered several tenants on the same
//! server each tenant opened its own connection. [`Resp3Pool`] turns
//! that around: a pool keyed on `(endpoint, auth)` hands out borrowed
//! connections from an idle queue and parks callers when the
//! `max_total` cap is hit.
//!
//! ## What it does
//!
//! - Bounded idle queue (`max_idle`, default 4) — connections beyond
//!   that count are closed instead of reused.
//! - Total in-flight cap (`max_total`, default 16) — callers past
//!   the cap block on a condvar until someone returns.
//! - Idle timeout (`idle_timeout`, default 60 s) — sockets that have
//!   been sitting unused are recycled rather than handed out, so an
//!   intermediate router that dropped state doesn't bite the next
//!   query.
//! - Drop-on-failure semantics — a connection that fails mid-call is
//!   not returned to the pool. The caller's error propagates; the
//!   next acquire opens a fresh socket.
//!
//! ## What it does NOT do
//!
//! - **No retries on the pool side.** The membrane already tolerates
//!   peer failures via its "log + skip" path; layering retries here
//!   would conflict with that.
//! - **No multiplexing.** Each [`PooledConnection`] is owned
//!   exclusively for the duration of the borrow. Concurrent queries
//!   to the same peer use distinct connections (subject to
//!   `max_total`), not pipelining over one.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use skeg_rigging_net::NetError;

use crate::connection::Resp3Connection;

/// Knobs for a [`Resp3Pool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Max idle connections kept in the pool. Returns past this cap
    /// are dropped (closing the socket) rather than queued.
    pub max_idle: usize,
    /// Max total connections (idle + in-flight) the pool will hand
    /// out. Callers past this cap block on [`Resp3Pool::acquire`]
    /// until a borrow is returned.
    pub max_total: usize,
    /// Connections idle longer than this are closed at acquire time
    /// rather than handed out.
    pub idle_timeout: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_idle: 4,
            max_total: 16,
            idle_timeout: Duration::from_secs(60),
        }
    }
}

/// Pool of RESP3 connections to one `(endpoint, auth)` target.
///
/// Cheap to clone — internally an `Arc`. Share one across the
/// `Resp3Tenant`s that talk to the same server.
#[derive(Clone)]
pub struct Resp3Pool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    endpoint: String,
    auth: Option<(String, String)>,
    config: PoolConfig,
    state: Mutex<PoolState>,
    not_at_cap: Condvar,
}

struct PoolState {
    /// Idle connections paired with the time they last returned.
    idle: VecDeque<(Resp3Connection, Instant)>,
    /// In-flight (borrowed) + idle count combined. Capped at
    /// `config.max_total`.
    in_use: usize,
}

impl Resp3Pool {
    /// Build a pool against `endpoint` with default knobs.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self::with_config(endpoint, None, PoolConfig::default())
    }

    /// Build a pool with explicit auth credentials.
    pub fn with_auth(endpoint: impl Into<String>, auth: (String, String)) -> Self {
        Self::with_config(endpoint, Some(auth), PoolConfig::default())
    }

    /// Build a pool with full configuration.
    pub fn with_config(
        endpoint: impl Into<String>,
        auth: Option<(String, String)>,
        config: PoolConfig,
    ) -> Self {
        Self {
            inner: Arc::new(PoolInner {
                endpoint: endpoint.into(),
                auth,
                config,
                state: Mutex::new(PoolState {
                    idle: VecDeque::new(),
                    in_use: 0,
                }),
                not_at_cap: Condvar::new(),
            }),
        }
    }

    /// Borrow a connection. Returns to the pool on `Drop`. Blocks if
    /// the pool is at `max_total` until someone releases.
    pub fn acquire(&self) -> Result<PooledConnection, NetError> {
        // Step 1: discard idle entries that have aged past
        // `idle_timeout` (and a fresh one will be opened below).
        // Step 2: if an idle entry remains, hand it out.
        // Step 3: if `in_use < max_total`, open a new one.
        // Step 4: otherwise wait on the condvar.
        let mut state = self.inner.state.lock().expect("pool mutex poisoned");
        loop {
            let timeout = self.inner.config.idle_timeout;
            while let Some(&(_, ts)) = state.idle.front() {
                if ts.elapsed() < timeout {
                    break;
                }
                // Idle too long: drop it (closes the socket).
                let _ = state.idle.pop_front();
            }
            if let Some((conn, _)) = state.idle.pop_front() {
                state.in_use += 1;
                return Ok(PooledConnection {
                    conn: Some(conn),
                    inner: self.inner.clone(),
                });
            }
            if state.in_use < self.inner.config.max_total {
                state.in_use += 1;
                // Drop the lock while we do the TCP connect.
                drop(state);
                let result = Resp3Connection::connect(
                    &self.inner.endpoint,
                    self.inner
                        .auth
                        .as_ref()
                        .map(|(u, p)| (u.as_str(), p.as_str())),
                );
                match result {
                    Ok(conn) => {
                        return Ok(PooledConnection {
                            conn: Some(conn),
                            inner: self.inner.clone(),
                        });
                    }
                    Err(e) => {
                        // Roll back the reservation: connect failed,
                        // we didn't actually hold a connection.
                        let mut state = self.inner.state.lock().expect("pool mutex poisoned");
                        state.in_use -= 1;
                        self.inner.not_at_cap.notify_one();
                        return Err(e);
                    }
                }
            }
            // At cap; wait for a release.
            state = self
                .inner
                .not_at_cap
                .wait(state)
                .expect("pool condvar poisoned");
        }
    }

    /// Current count of idle (warm, not borrowed) connections.
    /// Useful for tests / metrics.
    pub fn idle_count(&self) -> usize {
        self.inner
            .state
            .lock()
            .expect("pool mutex poisoned")
            .idle
            .len()
    }

    /// Current count of in-use (borrowed or idle) connections. Mirrors
    /// the `max_total` budget: `in_use_count() == max_total` means
    /// the next acquire will block.
    pub fn in_use_count(&self) -> usize {
        self.inner.state.lock().expect("pool mutex poisoned").in_use
    }
}

/// RAII guard for a borrowed connection. Returns to the pool on
/// `Drop` (or closes if the pool is full or the connection failed
/// via [`PooledConnection::discard`]).
pub struct PooledConnection {
    conn: Option<Resp3Connection>,
    inner: Arc<PoolInner>,
}

impl PooledConnection {
    /// Mutable borrow of the underlying connection. Use this to issue
    /// commands; the borrow ends when `self` drops.
    pub fn conn_mut(&mut self) -> &mut Resp3Connection {
        self.conn
            .as_mut()
            .expect("PooledConnection used after discard")
    }

    /// Drop the connection without returning it to the pool. Call
    /// this after a command failure that leaves the socket in an
    /// undefined state. The pool decrements its `in_use` counter so
    /// the next acquire can open a fresh connection.
    pub fn discard(mut self) {
        self.conn = None;
        // Drop fires below and recycles the counter.
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        let mut state = self.inner.state.lock().expect("pool mutex poisoned");
        state.in_use -= 1;
        if let Some(conn) = self.conn.take()
            && state.idle.len() < self.inner.config.max_idle
        {
            state.idle.push_back((conn, Instant::now()));
        }
        // Either we returned the connection or dropped it (in_use
        // already decremented). Notify a waiter.
        self.inner.not_at_cap.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    /// Tiny RESP3 echo of HELLO so `Resp3Connection::connect` succeeds.
    /// We don't need to support any other command for these tests.
    fn spawn_hello_server() -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let connects = Arc::new(AtomicUsize::new(0));
        let counter = connects.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                counter.fetch_add(1, Ordering::SeqCst);
                thread::spawn(move || {
                    use std::io::{Read, Write};
                    // Read HELLO (we just drain whatever bytes arrive
                    // until we have enough to consider it a HELLO).
                    let mut buf = [0u8; 256];
                    let _ = s.read(&mut buf);
                    // Reply with an empty RESP3 Map: `%0\r\n`.
                    let _ = s.write_all(b"%0\r\n");
                    // Keep socket open for the test; close on read = 0.
                    loop {
                        match s.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {}
                        }
                    }
                });
            }
        });
        (endpoint, connects)
    }

    #[test]
    fn acquire_and_return_reuses_connection() {
        let (endpoint, connects) = spawn_hello_server();
        let pool = Resp3Pool::new(endpoint);
        {
            let _c1 = pool.acquire().unwrap();
            assert_eq!(pool.in_use_count(), 1);
            assert_eq!(pool.idle_count(), 0);
        }
        assert_eq!(pool.in_use_count(), 0);
        assert_eq!(pool.idle_count(), 1);

        // Second acquire must NOT open a new socket; it should reuse
        // the idle one.
        {
            let _c2 = pool.acquire().unwrap();
        }
        // Server saw exactly one connect.
        assert_eq!(connects.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn max_idle_caps_returned_count() {
        let (endpoint, _) = spawn_hello_server();
        let pool = Resp3Pool::with_config(
            endpoint,
            None,
            PoolConfig {
                max_idle: 1,
                max_total: 4,
                idle_timeout: Duration::from_secs(60),
            },
        );
        // Acquire 3 concurrent; release them in order. Only 1 should
        // be retained.
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        let c = pool.acquire().unwrap();
        drop(a);
        drop(b);
        drop(c);
        assert_eq!(pool.idle_count(), 1);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn max_total_blocks_until_released() {
        let (endpoint, _) = spawn_hello_server();
        let pool = Resp3Pool::with_config(
            endpoint,
            None,
            PoolConfig {
                max_idle: 4,
                max_total: 2,
                idle_timeout: Duration::from_secs(60),
            },
        );
        let a = pool.acquire().unwrap();
        let b = pool.acquire().unwrap();
        assert_eq!(pool.in_use_count(), 2);

        let pool2 = pool.clone();
        let handle = thread::spawn(move || {
            // This blocks until something is returned.
            let _c = pool2.acquire().unwrap();
        });
        thread::sleep(Duration::from_millis(50));
        // Worker still parked.
        assert!(!handle.is_finished());
        drop(a);
        handle.join().unwrap();
        drop(b);
    }

    #[test]
    fn discard_does_not_return_to_pool() {
        let (endpoint, connects) = spawn_hello_server();
        let pool = Resp3Pool::new(endpoint);
        let c = pool.acquire().unwrap();
        c.discard();
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.in_use_count(), 0);
        // Next acquire must open a fresh socket.
        let _c2 = pool.acquire().unwrap();
        assert_eq!(connects.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn idle_timeout_recycles_stale_connection() {
        let (endpoint, connects) = spawn_hello_server();
        let pool = Resp3Pool::with_config(
            endpoint,
            None,
            PoolConfig {
                max_idle: 4,
                max_total: 4,
                idle_timeout: Duration::from_millis(10),
            },
        );
        drop(pool.acquire().unwrap());
        assert_eq!(pool.idle_count(), 1);
        thread::sleep(Duration::from_millis(30));
        // After idle_timeout the entry is discarded on the next acquire.
        let _c = pool.acquire().unwrap();
        assert_eq!(connects.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn connect_failure_releases_reservation() {
        // No server running on this port — acquire must roll back.
        let pool = Resp3Pool::new("127.0.0.1:1"); // port 1 should reject
        let result = pool.acquire();
        assert!(result.is_err());
        assert_eq!(pool.in_use_count(), 0);
    }
}
