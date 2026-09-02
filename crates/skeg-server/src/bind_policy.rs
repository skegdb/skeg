//! Refuse to bind a non-loopback address without authentication.
//!
//! The single-tenant binaries (`skeg`, `skeg-resp3`) have no auth layer at
//! all - anyone who can reach the listening port can read, write and drop
//! indices. Binding loopback is always fine (only local processes can
//! reach it); binding anything else - including the commonly-used
//! `0.0.0.0` / `::` "all interfaces" address - is refused unless the
//! operator explicitly opts in with `--allow-unauthenticated-network`
//! (env `SKEG_ALLOW_UNAUTHENTICATED_NETWORK`).

/// The flag/env pair named in the refusal message, kept in one place so
/// the CLI help text and the error message can't drift apart.
pub const ALLOW_FLAG: &str = "--allow-unauthenticated-network";
pub const ALLOW_ENV: &str = "SKEG_ALLOW_UNAUTHENTICATED_NETWORK";

/// Check whether `addr` may be bound given the current `allow_network`
/// opt-in.
///
/// - `addr` is first tried as a `SocketAddr` literal. When that fails (e.g.
///   a hostname like `localhost:7379`, or a non-canonical numeric form like
///   `0:7379` or `0x00000000:7379`, both of which `tokio::net::ToSocketAddrs`
///   happily resolves and the real bind call would accept), it falls back
///   to `std::net::ToSocketAddrs` resolution so those forms can't slip past
///   the loopback check. If resolution also fails, returns `Ok(false)` -
///   the real bind call will report the error with better context than
///   this pure check could.
/// - The loopback test is applied to every resolved address: if any of
///   them is non-loopback, the whole `addr` is treated as non-loopback.
///   Note this is conservative for IPv4-mapped IPv6 addresses:
///   `[::ffff:127.0.0.1]` is refused (without the opt-in) even though it
///   maps to a loopback IPv4 address, because `Ipv6Addr::is_loopback`
///   returns `false` for it.
/// - All-loopback resolutions always return `Ok(false)`: no opt-in was
///   needed, so the caller should not warn.
/// - Otherwise, returns `Ok(true)` when `allow_network` is `true` - the
///   bind is allowed, but the caller should warn that it is
///   unauthenticated - otherwise `Err` with an operator-facing message.
pub fn check_unauthenticated_bind(addr: &str, allow_network: bool) -> Result<bool, String> {
    let resolved: Vec<std::net::SocketAddr> = match addr.parse::<std::net::SocketAddr>() {
        Ok(sock_addr) => vec![sock_addr],
        Err(_) => {
            use std::net::ToSocketAddrs;
            match addr.to_socket_addrs() {
                Ok(iter) => iter.collect(),
                Err(_) => return Ok(false),
            }
        }
    };
    if resolved.iter().all(|a| a.ip().is_loopback()) {
        return Ok(false);
    }
    if allow_network {
        return Ok(true);
    }
    Err(format!(
        "refusing to bind {addr}: this server has no authentication - anyone who can reach \
         this address can read, write and drop indices. Bind a loopback address (e.g. \
         127.0.0.1 or ::1) instead, put an authenticating proxy or `skeg-server-tenant` in \
         front of it, or pass {ALLOW_FLAG} (env {ALLOW_ENV}=1) to accept the risk."
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_v4_is_always_ok() {
        assert_eq!(
            check_unauthenticated_bind("127.0.0.1:7379", false),
            Ok(false)
        );
        assert_eq!(
            check_unauthenticated_bind("127.0.0.1:7379", true),
            Ok(false)
        );
    }

    #[test]
    fn loopback_v6_is_always_ok() {
        assert_eq!(check_unauthenticated_bind("[::1]:7379", false), Ok(false));
        assert_eq!(check_unauthenticated_bind("[::1]:7379", true), Ok(false));
    }

    #[test]
    fn unspecified_v4_is_refused_without_flag() {
        assert!(check_unauthenticated_bind("0.0.0.0:7379", false).is_err());
    }

    #[test]
    fn unspecified_v4_is_ok_with_flag() {
        assert_eq!(check_unauthenticated_bind("0.0.0.0:7379", true), Ok(true));
    }

    #[test]
    fn unspecified_v6_is_refused_without_flag() {
        assert!(check_unauthenticated_bind("[::]:7379", false).is_err());
    }

    #[test]
    fn unspecified_v6_is_ok_with_flag() {
        assert_eq!(check_unauthenticated_bind("[::]:7379", true), Ok(true));
    }

    #[test]
    fn private_network_address_is_refused_without_flag() {
        assert!(check_unauthenticated_bind("10.0.0.5:7379", false).is_err());
        assert_eq!(check_unauthenticated_bind("10.0.0.5:7379", true), Ok(true));
    }

    #[test]
    fn network_address_with_port_zero_is_refused_without_flag() {
        assert!(check_unauthenticated_bind("192.168.1.2:0", false).is_err());
        assert_eq!(check_unauthenticated_bind("192.168.1.2:0", true), Ok(true));
    }

    #[test]
    fn unparseable_addr_is_deferred_to_bind() {
        assert_eq!(check_unauthenticated_bind("not-an-addr", false), Ok(false));
        assert_eq!(check_unauthenticated_bind("not-an-addr", true), Ok(false));
    }

    #[test]
    fn hostname_loopback_is_ok() {
        // "localhost" isn't a `SocketAddr` literal, so this exercises the
        // `ToSocketAddrs` resolution fallback; every resolved address
        // (127.0.0.1 and/or ::1) is loopback.
        assert_eq!(
            check_unauthenticated_bind("localhost:7379", false),
            Ok(false)
        );
    }

    #[test]
    fn non_canonical_unspecified_form_is_refused_without_flag() {
        // "0" is not a `SocketAddr` literal but resolves (via getaddrinfo)
        // to 0.0.0.0 - the same non-loopback "all interfaces" address as
        // "0.0.0.0", just spelled differently. Must not bypass the check.
        assert!(check_unauthenticated_bind("0:7379", false).is_err());
    }

    #[test]
    fn non_canonical_unspecified_form_is_ok_with_flag() {
        assert_eq!(check_unauthenticated_bind("0:7379", true), Ok(true));
    }

    #[test]
    fn error_message_names_address_and_flag() {
        let err = check_unauthenticated_bind("10.0.0.5:7379", false).unwrap_err();
        assert!(err.contains("10.0.0.5:7379"), "message: {err}");
        assert!(err.contains(ALLOW_FLAG), "message: {err}");
        assert!(err.contains(ALLOW_ENV), "message: {err}");
    }
}
