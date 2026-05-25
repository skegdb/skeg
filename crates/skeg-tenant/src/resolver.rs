//! Tenant resolution.
//!
//! Resolvers map a request to a `TenantId`. Three concrete strategies plus
//! a chain that tries them in order:
//!
//! - `AuthBoundResolver`: tenant is fixed for the lifetime of a connection
//!   once the client has authenticated. This is the safest default and the
//!   one we recommend for RESP3 HELLO AUTH flows.
//! - `PerRequestResolver`: tenant id arrives in the request itself
//!   (extended binary header). Useful for gateways that multiplex many
//!   tenants over a single backend connection.
//! - `ConnectionBoundResolver`: tenant derived from connection metadata
//!   (TCP source, SNI hostname). Useful when fronted by a reverse proxy
//!   that already terminates per-tenant identity.
//! - `NullResolver`: always `ZERO`. Wired in single-tenant deployments
//!   and exercised by every existing test (so the back-compat path is the
//!   one we walk on by default).
//!
//! NOTE: `ResolverChain` is ordered. Put strict resolvers first.
//!
//! `ResolverChain` evaluates in order and returns the first successful
//! resolution. A chain `[AuthBound, PerRequest]` lets a connection that
//! has authenticated keep its identity, while still allowing an
//! unauthenticated gateway to assert per-request tenancy.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::id::TenantId;

/// What every resolver can fail with.
#[derive(Debug, thiserror::Error, PartialEq, Eq, Clone)]
pub enum ResolveError {
    /// The resolver has no opinion (try the next link in the chain).
    #[error("resolver abstained")]
    Abstain,
    /// Required identity material was missing.
    #[error("missing tenant identity")]
    Missing,
    /// Identity material was present but invalid.
    #[error("invalid tenant identity: {0}")]
    Invalid(String),
}

/// What a resolver can see at decision time.
///
/// `auth_tenant` is set once a connection has completed authentication
/// (the `AuthBoundResolver` reads it from here). `peer` carries the TCP
/// source address. `hint` carries the per-request id from an extended
/// header if the wire protocol supplies one.
#[derive(Debug, Default, Clone)]
pub struct ResolveContext {
    pub auth_tenant: Option<TenantId>,
    pub peer: Option<std::net::SocketAddr>,
    pub sni: Option<String>,
    pub hint: Option<TenantId>,
}

impl ResolveContext {
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_auth(mut self, t: TenantId) -> Self {
        self.auth_tenant = Some(t);
        self
    }

    #[must_use]
    pub fn with_hint(mut self, t: TenantId) -> Self {
        self.hint = Some(t);
        self
    }

    #[must_use]
    pub fn with_peer(mut self, addr: std::net::SocketAddr) -> Self {
        self.peer = Some(addr);
        self
    }

    #[must_use]
    pub fn with_sni(mut self, sni: String) -> Self {
        self.sni = Some(sni);
        self
    }
}

/// Trait every resolver implements. Async-ready for future
/// remote-lookup resolvers, but the four canonical ones are sync.
pub trait TenantResolver: Send + Sync {
    /// Resolve a `TenantId` from the context, or fail.
    ///
    /// # Errors
    ///
    /// Returns `ResolveError::Abstain` to let a chain fall through to the
    /// next resolver. Returns `ResolveError::Missing` or `Invalid` to
    /// stop the chain with a definite failure.
    fn resolve(&self, ctx: &ResolveContext) -> Result<TenantId, ResolveError>;
}

/// Always returns `TenantId::ZERO`. Single-tenant default.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullResolver;

impl TenantResolver for NullResolver {
    fn resolve(&self, _ctx: &ResolveContext) -> Result<TenantId, ResolveError> {
        Ok(TenantId::ZERO)
    }
}

/// Uses the tenant identity stamped on the connection at AUTH time.
#[derive(Debug, Default, Clone, Copy)]
pub struct AuthBoundResolver;

impl TenantResolver for AuthBoundResolver {
    fn resolve(&self, ctx: &ResolveContext) -> Result<TenantId, ResolveError> {
        ctx.auth_tenant.ok_or(ResolveError::Abstain)
    }
}

/// Reads the tenant id from the per-request hint.
#[derive(Debug, Default, Clone, Copy)]
pub struct PerRequestResolver;

impl TenantResolver for PerRequestResolver {
    fn resolve(&self, ctx: &ResolveContext) -> Result<TenantId, ResolveError> {
        ctx.hint.ok_or(ResolveError::Abstain)
    }
}

/// Maps connection metadata (SNI hostname or peer address) to a tenant.
/// The mapping is held under an `RwLock` so an admin endpoint can add
/// or remove bindings at runtime without restarting the server.
#[derive(Debug, Default, Clone)]
pub struct ConnectionBoundResolver {
    inner: Arc<RwLock<ConnectionBindings>>,
}

#[derive(Debug, Default)]
struct ConnectionBindings {
    by_sni: ahash::AHashMap<String, TenantId>,
    by_peer_ip: ahash::AHashMap<std::net::IpAddr, TenantId>,
}

impl ConnectionBoundResolver {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind_sni(&self, sni: impl Into<String>, t: TenantId) {
        self.inner.write().by_sni.insert(sni.into(), t);
    }

    pub fn bind_peer(&self, ip: std::net::IpAddr, t: TenantId) {
        self.inner.write().by_peer_ip.insert(ip, t);
    }

    #[must_use]
    pub fn unbind_sni(&self, sni: &str) -> Option<TenantId> {
        self.inner.write().by_sni.remove(sni)
    }
}

impl TenantResolver for ConnectionBoundResolver {
    fn resolve(&self, ctx: &ResolveContext) -> Result<TenantId, ResolveError> {
        let guard = self.inner.read();
        if let Some(sni) = ctx.sni.as_deref()
            && let Some(t) = guard.by_sni.get(sni)
        {
            return Ok(*t);
        }
        if let Some(peer) = ctx.peer
            && let Some(t) = guard.by_peer_ip.get(&peer.ip())
        {
            return Ok(*t);
        }
        Err(ResolveError::Abstain)
    }
}

/// Try a sequence of resolvers in order. First non-`Abstain` answer wins.
/// If every resolver abstains, the chain falls back to `ZERO`.
pub struct ResolverChain {
    chain: Vec<Box<dyn TenantResolver>>,
    fallback_to_zero: bool,
}

impl ResolverChain {
    #[must_use]
    pub fn new() -> Self {
        Self {
            chain: Vec::new(),
            fallback_to_zero: true,
        }
    }

    /// Disable the fallback. With this set, a fully-abstaining chain
    /// returns `Missing` instead of `ZERO`. Multi-tenant deployments
    /// that must reject unauthenticated traffic want this.
    #[must_use]
    pub fn strict(mut self) -> Self {
        self.fallback_to_zero = false;
        self
    }

    #[must_use]
    pub fn with<R: TenantResolver + 'static>(mut self, r: R) -> Self {
        self.chain.push(Box::new(r));
        self
    }
}

impl Default for ResolverChain {
    fn default() -> Self {
        Self::new()
    }
}

impl TenantResolver for ResolverChain {
    fn resolve(&self, ctx: &ResolveContext) -> Result<TenantId, ResolveError> {
        for r in &self.chain {
            match r.resolve(ctx) {
                Ok(t) => return Ok(t),
                Err(ResolveError::Abstain) => {}
                Err(other) => return Err(other),
            }
        }
        if self.fallback_to_zero {
            Ok(TenantId::ZERO)
        } else {
            Err(ResolveError::Missing)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn ip(v: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, v))
    }

    #[test]
    fn null_resolver_always_zero() {
        let r = NullResolver;
        let t = r.resolve(&ResolveContext::empty()).unwrap();
        assert_eq!(t, TenantId::ZERO);
    }

    #[test]
    fn auth_bound_reads_auth_tenant_field() {
        let id = TenantId::from_name("alice");
        let r = AuthBoundResolver;
        let t = r.resolve(&ResolveContext::empty().with_auth(id)).unwrap();
        assert_eq!(t, id);
    }

    #[test]
    fn auth_bound_abstains_without_auth() {
        let r = AuthBoundResolver;
        let e = r.resolve(&ResolveContext::empty()).unwrap_err();
        assert_eq!(e, ResolveError::Abstain);
    }

    #[test]
    fn per_request_reads_hint() {
        let id = TenantId::from_name("bob");
        let r = PerRequestResolver;
        let t = r.resolve(&ResolveContext::empty().with_hint(id)).unwrap();
        assert_eq!(t, id);
    }

    #[test]
    fn connection_bound_by_sni() {
        let r = ConnectionBoundResolver::new();
        let alice = TenantId::from_name("alice");
        r.bind_sni("alice.skeg.example", alice);

        let ctx = ResolveContext::empty().with_sni("alice.skeg.example".into());
        assert_eq!(r.resolve(&ctx).unwrap(), alice);

        let unknown = ResolveContext::empty().with_sni("nobody.example".into());
        assert_eq!(r.resolve(&unknown).unwrap_err(), ResolveError::Abstain);
    }

    #[test]
    fn connection_bound_by_peer_ip() {
        let r = ConnectionBoundResolver::new();
        let dave = TenantId::from_name("dave");
        r.bind_peer(ip(7), dave);

        let ctx = ResolveContext::empty().with_peer(SocketAddr::new(ip(7), 9001));
        assert_eq!(r.resolve(&ctx).unwrap(), dave);
    }

    #[test]
    fn chain_first_non_abstain_wins() {
        let alice = TenantId::from_name("alice");
        let chain = ResolverChain::new()
            .with(AuthBoundResolver)
            .with(PerRequestResolver);

        // Both provide a value, AuthBound is first.
        let ctx = ResolveContext::empty()
            .with_auth(alice)
            .with_hint(TenantId::from_name("bob"));
        assert_eq!(chain.resolve(&ctx).unwrap(), alice);

        // Auth abstains, hint wins.
        let bob = TenantId::from_name("bob");
        let ctx2 = ResolveContext::empty().with_hint(bob);
        assert_eq!(chain.resolve(&ctx2).unwrap(), bob);
    }

    #[test]
    fn chain_falls_back_to_zero_when_lenient() {
        let chain = ResolverChain::new()
            .with(AuthBoundResolver)
            .with(PerRequestResolver);
        let t = chain.resolve(&ResolveContext::empty()).unwrap();
        assert_eq!(t, TenantId::ZERO);
    }

    #[test]
    fn chain_strict_rejects_anonymous() {
        let chain = ResolverChain::new()
            .with(AuthBoundResolver)
            .with(PerRequestResolver)
            .strict();
        let e = chain.resolve(&ResolveContext::empty()).unwrap_err();
        assert_eq!(e, ResolveError::Missing);
    }

    #[test]
    fn definite_error_short_circuits_chain() {
        struct Boom;
        impl TenantResolver for Boom {
            fn resolve(&self, _: &ResolveContext) -> Result<TenantId, ResolveError> {
                Err(ResolveError::Invalid("boom".into()))
            }
        }
        let chain = ResolverChain::new().with(Boom).with(PerRequestResolver);
        let bob = TenantId::from_name("bob");
        let ctx = ResolveContext::empty().with_hint(bob);
        // PerRequest would have answered bob, but Boom's hard error stops us.
        let e = chain.resolve(&ctx).unwrap_err();
        matches!(e, ResolveError::Invalid(_));
    }
}
