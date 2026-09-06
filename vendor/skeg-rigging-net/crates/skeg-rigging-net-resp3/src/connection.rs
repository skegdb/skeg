//! Synchronous RESP3 connection backed by `std::net::TcpStream`.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use skeg_resp3::{Frame, FrameDecoder, ProtoVersion, encode_frame};
use skeg_rigging_net::NetError;

/// Synchronous RESP3 connection to a skeg-server.
///
/// Calls are request-response and **not** thread-safe: each connection
/// belongs to one logical caller. Use a pool (or wrap in a Mutex) for
/// shared access.
pub struct Resp3Connection {
    stream: TcpStream,
    decoder: FrameDecoder,
    read_buf: [u8; 8192],
}

impl Resp3Connection {
    /// Connect to `endpoint` (`host:port`) and run `HELLO 3` optionally
    /// with `AUTH user pass`. Sets a default read timeout of 5 seconds.
    pub fn connect(endpoint: &str, auth: Option<(&str, &str)>) -> Result<Self, NetError> {
        let stream = TcpStream::connect(endpoint)?;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        stream.set_nodelay(true)?;
        let mut conn = Self {
            stream,
            decoder: FrameDecoder::new(),
            read_buf: [0u8; 8192],
        };
        conn.hello(3, auth)?;
        Ok(conn)
    }

    /// Connect using an already-established stream. Primarily for
    /// tests using a loopback mock server.
    pub fn from_stream(stream: TcpStream, auth: Option<(&str, &str)>) -> Result<Self, NetError> {
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        stream.set_nodelay(true)?;
        let mut conn = Self {
            stream,
            decoder: FrameDecoder::new(),
            read_buf: [0u8; 8192],
        };
        conn.hello(3, auth)?;
        Ok(conn)
    }

    fn hello(&mut self, version: u8, auth: Option<(&str, &str)>) -> Result<(), NetError> {
        let mut args: Vec<Frame> = vec![bulk_str("HELLO"), bulk_str(&version.to_string())];
        if let Some((u, p)) = auth {
            args.push(bulk_str("AUTH"));
            args.push(bulk_str(u));
            args.push(bulk_str(p));
        }
        self.send(&Frame::Array(args))?;
        match self.recv()? {
            // RESP3 HELLO reply is a Map of metadata.
            Frame::Map(_) | Frame::Array(_) => Ok(()),
            Frame::Error(e) => Err(NetError::Auth(e)),
            other => Err(NetError::Protocol(format!(
                "unexpected HELLO reply: {other:?}"
            ))),
        }
    }

    /// Send a raw frame.
    pub fn send(&mut self, frame: &Frame) -> Result<(), NetError> {
        let mut buf = BytesMut::new();
        encode_frame(frame, ProtoVersion::Resp3, &mut buf);
        self.stream.write_all(&buf)?;
        Ok(())
    }

    /// Receive one frame, blocking until it is complete or the socket
    /// closes.
    pub fn recv(&mut self) -> Result<Frame, NetError> {
        loop {
            match self.decoder.decode() {
                Ok(Some(frame)) => return Ok(frame),
                Ok(None) => {} // need more bytes
                Err(e) => {
                    return Err(NetError::Protocol(format!("parse error: {e}")));
                }
            }
            let n = self.stream.read(&mut self.read_buf)?;
            if n == 0 {
                return Err(NetError::Protocol("connection closed mid-frame".into()));
            }
            self.decoder.feed(&self.read_buf[..n]);
        }
    }

    /// Convenience: send `cmd args...` and read the single reply.
    pub fn call(&mut self, cmd: &str, args: &[Bytes]) -> Result<Frame, NetError> {
        let mut parts: Vec<Frame> = Vec::with_capacity(args.len() + 1);
        parts.push(bulk_str(cmd));
        for a in args {
            parts.push(Frame::Bulk(a.clone()));
        }
        self.send(&Frame::Array(parts))?;
        let frame = self.recv()?;
        if let Frame::Error(e) = &frame {
            return Err(NetError::Remote(e.clone()));
        }
        Ok(frame)
    }
}

/// Build a `Frame::Bulk` from a string slice.
pub(crate) fn bulk_str(s: &str) -> Frame {
    Frame::Bulk(Bytes::copy_from_slice(s.as_bytes()))
}

/// Encode a `Vec<f32>` as little-endian bytes for transport.
pub(crate) fn encode_vector(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
