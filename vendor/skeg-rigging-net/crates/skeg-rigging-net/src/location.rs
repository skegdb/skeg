//! How a tenant can be addressed across transports.
//!
//! `TenantLocation` is the discriminator the membrane uses to decide
//! which `PeerOpener` to invoke for a given member. Hansa's
//! `MemberRecord` carries one of these in place of the v0.1
//! `tenant_path: PathBuf`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::NetError;

/// Tenant address. Variants will grow as transports come online.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum TenantLocation {
    /// Local filesystem path (the existing in-process adapter case).
    Path {
        /// Tenant directory on disk.
        path: PathBuf,
    },
    /// Reachable via RESP3 at `host:port`. The tenant's index name in
    /// skeg-server is `hansa` by convention.
    Resp3 {
        /// `host:port` of the skeg-server.
        endpoint: String,
        /// Optional `(username, password)` for `HELLO 3 AUTH`.
        #[serde(default)]
        auth: Option<(String, String)>,
    },
    /// Reachable via HTTP at `base_url`. Reserved for the upcoming
    /// `skeg-rigging-net-http` transport.
    Http {
        /// Base URL like `https://host:port/`.
        base_url: String,
        /// Optional bearer token.
        #[serde(default)]
        bearer: Option<String>,
    },
}

/// Parse a `TenantLocation` from a URL-like string:
///
/// - `file:///path/to/dir` or plain path → [`TenantLocation::Path`]
/// - `resp3://host:port` (with optional `user:pass@` prefix) → [`TenantLocation::Resp3`]
/// - `http://host:port/` or `https://...` → [`TenantLocation::Http`]
///
/// This is a deliberately small parser; callers building locations
/// programmatically should construct the enum directly.
pub fn parse_location(s: &str) -> Result<TenantLocation, NetError> {
    if let Some(rest) = s.strip_prefix("file://") {
        return Ok(TenantLocation::Path {
            path: PathBuf::from(rest),
        });
    }
    if let Some(rest) = s.strip_prefix("resp3://") {
        let (auth, hostport) = if let Some(at) = rest.rfind('@') {
            let (creds, hp) = rest.split_at(at);
            let hp = &hp[1..];
            if let Some((u, p)) = creds.split_once(':') {
                (Some((u.to_string(), p.to_string())), hp.to_string())
            } else {
                return Err(NetError::BadLocation(format!(
                    "resp3 credentials must be user:pass, got {creds:?}"
                )));
            }
        } else {
            (None, rest.to_string())
        };
        if hostport.is_empty() {
            return Err(NetError::BadLocation("resp3:// missing host:port".into()));
        }
        return Ok(TenantLocation::Resp3 {
            endpoint: hostport,
            auth,
        });
    }
    if s.starts_with("http://") || s.starts_with("https://") {
        return Ok(TenantLocation::Http {
            base_url: s.to_string(),
            bearer: None,
        });
    }
    // No recognised scheme → treat as a bare filesystem path.
    if s.contains("://") {
        return Err(NetError::BadLocation(format!("unknown scheme in {s:?}")));
    }
    Ok(TenantLocation::Path {
        path: PathBuf::from(s),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_path() {
        let loc = parse_location("/tmp/tenant-1").unwrap();
        assert_eq!(
            loc,
            TenantLocation::Path {
                path: PathBuf::from("/tmp/tenant-1")
            }
        );
    }

    #[test]
    fn parses_file_url() {
        let loc = parse_location("file:///var/lib/tenant-2").unwrap();
        assert!(matches!(loc, TenantLocation::Path { .. }));
    }

    #[test]
    fn parses_resp3_no_auth() {
        let loc = parse_location("resp3://example.org:6379").unwrap();
        match loc {
            TenantLocation::Resp3 { endpoint, auth } => {
                assert_eq!(endpoint, "example.org:6379");
                assert!(auth.is_none());
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn parses_resp3_with_auth() {
        let loc = parse_location("resp3://alice:secret@h:6379").unwrap();
        match loc {
            TenantLocation::Resp3 { endpoint, auth } => {
                assert_eq!(endpoint, "h:6379");
                assert_eq!(auth, Some(("alice".to_string(), "secret".to_string())));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn parses_http() {
        let loc = parse_location("https://h:443/").unwrap();
        assert!(matches!(loc, TenantLocation::Http { .. }));
    }

    #[test]
    fn rejects_unknown_scheme() {
        let err = parse_location("ftp://x").unwrap_err();
        assert!(matches!(err, NetError::BadLocation(_)));
    }

    #[test]
    fn json_roundtrip_path() {
        let loc = TenantLocation::Path {
            path: PathBuf::from("/x"),
        };
        let s = serde_json::to_string(&loc).unwrap();
        let back: TenantLocation = serde_json::from_str(&s).unwrap();
        assert_eq!(loc, back);
    }

    #[test]
    fn json_roundtrip_resp3() {
        let loc = TenantLocation::Resp3 {
            endpoint: "h:6379".into(),
            auth: Some(("u".into(), "p".into())),
        };
        let s = serde_json::to_string(&loc).unwrap();
        let back: TenantLocation = serde_json::from_str(&s).unwrap();
        assert_eq!(loc, back);
    }
}
