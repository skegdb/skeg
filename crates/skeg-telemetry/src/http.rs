//! Tiny HTTP exporter for `/metrics`.
//!
//! Runs on a dedicated thread that owns one `tiny_http::Server`. The
//! main process is never blocked; the exporter only reads the static
//! counters via [`crate::stats::dump_text`] when it gets a request.
//!
//! Opt-in: requires the `http` cargo feature. When the feature is off
//! this module is not compiled in at all.

use std::io::Cursor;
use std::net::SocketAddr;
use std::thread;

use tiny_http::{Header, Method, Response, Server};

use crate::stats;

/// Bind to `addr` and serve `/metrics` (and a small landing page on `/`).
///
/// This call blocks the calling thread. Callers that want it async should
/// pass it to a worker thread, e.g.:
///
/// ```ignore
/// std::thread::spawn(|| skeg_telemetry::http::serve_blocking("127.0.0.1:9090".parse().unwrap()));
/// ```
pub fn serve_blocking(addr: SocketAddr) -> std::io::Result<()> {
    let server = Server::http(addr).map_err(|e| std::io::Error::other(e.to_string()))?;
    let ct_metrics =
        Header::from_bytes(&b"Content-Type"[..], &b"text/plain; version=0.0.4"[..])
            .expect("header is valid ASCII");
    let ct_html = Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..])
        .expect("header is valid ASCII");

    for request in server.incoming_requests() {
        let path = request.url().to_string();
        match (request.method(), path.as_str()) {
            (Method::Get, "/metrics") => {
                let body = stats::dump_text();
                let resp = Response::new(
                    200.into(),
                    vec![ct_metrics.clone()],
                    Cursor::new(body.into_bytes()),
                    None,
                    None,
                );
                let _ = request.respond(resp);
            }
            (Method::Get, "/") => {
                let resp = Response::from_string(LANDING).with_header(ct_html.clone());
                let _ = request.respond(resp);
            }
            _ => {
                let _ = request.respond(Response::from_string("404\n").with_status_code(404));
            }
        }
    }
    Ok(())
}

/// Convenience: spawn `serve_blocking` on a named thread and return the
/// join handle.
pub fn spawn(addr: SocketAddr) -> std::io::Result<thread::JoinHandle<std::io::Result<()>>> {
    thread::Builder::new()
        .name("skeg-telemetry-http".into())
        .spawn(move || serve_blocking(addr))
}

const LANDING: &str = r#"<!doctype html>
<html><head><title>skeg metrics</title></head>
<body style="font-family: ui-monospace, monospace; padding: 24px;">
<h2>skeg telemetry</h2>
<p>Prometheus metrics at <a href="/metrics">/metrics</a>.</p>
</body></html>
"#;
