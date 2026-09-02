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
/// - If `addr` fails to parse as a `SocketAddr`, returns `Ok(false)` - the
///   real bind call will report the parse/connect error with better
///   context than this pure check could.
/// - Loopback addresses (`127.0.0.1`, `::1`, ...) always return `Ok(false)`:
///   no opt-in was needed, so the caller should not warn.
/// - Any other address (including the unspecified `0.0.0.0` / `::`, which
///   is not loopback) returns `Ok(true)` when `allow_network` is `true` -
///   the bind is allowed, but the caller should warn that it is
///   unauthenticated - otherwise `Err` with an operator-facing message.
pub fn check_unauthenticated_bind(addr: &str, allow_network: bool) -> Result<bool, String> {
    let Ok(sock_addr) = addr.parse::<std::net::SocketAddr>() else {
        return Ok(false);
    };
    if sock_addr.ip().is_loopback() {
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
    fn error_message_names_address_and_flag() {
        let err = check_unauthenticated_bind("10.0.0.5:7379", false).unwrap_err();
        assert!(err.contains("10.0.0.5:7379"), "message: {err}");
        assert!(err.contains(ALLOW_FLAG), "message: {err}");
        assert!(err.contains(ALLOW_ENV), "message: {err}");
    }
}
